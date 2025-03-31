import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import *

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

# Channel assignment (GAT)
def channel_assignment(uav_features, num_channels=NUM_CHANNELS, num_uavs=NUM_UAVS, epochs=200, lr=0.01):
    gat_model = GAT(input_dim=2, hidden_dim=8, output_dim=num_channels)
    optimizer_net = optim.Adam(gat_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    target_channels = torch.tensor([i % num_channels for i in range(num_uavs)], dtype=torch.long)
    for epoch in range(epochs):
        optimizer_net.zero_grad()
        output = gat_model(uav_features)
        loss = loss_fn(output, target_channels)
        loss.backward()
        optimizer_net.step()
        if epoch % 50 == 0:
            print(f"GAT Epoch {epoch}, Loss: {loss.item():.4f}")
    channels_assigned = torch.argmax(gat_model(uav_features), dim=1).numpy()
    return channels_assigned, gat_model

# def calculate_users_rates_per_channel(user_transmission_powers_one_channel,
#                                       path_losses, user_bs_associations,
#                                       consider_interference=True):
#     num_users = user_transmission_powers_one_channel.shape[0]
#     # desired signal: 每个 UAV 在其基站处的接收功率
#     desired_signals = np.array([user_transmission_powers_one_channel[i] * path_losses[i, user_bs_associations[i]]
#                                 for i in range(num_users)])
#     interference = np.zeros(num_users)
#     if consider_interference:
#         for i in range(num_users):
#             # 对于 UAV i，其所在的 BS 为 user_bs_associations[i]，将所有 UAV（包括不属于该 BS 的）对该 BS 的贡献累加
#             total_received = np.sum(user_transmission_powers_one_channel * path_losses[:, user_bs_associations[i]])
#             # 干扰为：总接收功率减去自身信号
#             interference[i] = total_received - desired_signals[i]
#     sinrs = desired_signals / (interference + NOISE)
#     rates = np.log2(1 + sinrs)
#     max_user_rate = 10
#     rates = np.clip(rates, 0, max_user_rate)
#     return rates, sinrs


# 根据各 UAV 到各基站的信道增益，计算基站分配（选择信道增益最大的基站）
def assign_base_stations(uav_positions, bs_locations):
    num_uavs = uav_positions.shape[0]
    num_bs = bs_locations.shape[0]
    path_losses = np.zeros((num_uavs, num_bs))
    for i in range(num_uavs):
        for j in range(num_bs):
            path_losses[i, j] = calculate_channel_gain(uav_positions[i], bs_locations[j])
    # 对每个 UAV，选择信道增益最大的基站
    associations = np.argmax(path_losses, axis=1)
    return path_losses, associations

def calculate_channel_gain(user_loc, bs_loc, f_c=FREQUENCY, sigma_dB=8):
    """
    采用自由空间路径损耗结合对数正态阴影衰落的模型计算信道增益。
    
    参数：
      user_loc: UAV二维位置 [x, y]，实际计算时会加入 UAV 高度（例如100m）
      bs_loc: 基站二维位置 [x, y]，实际计算时会加入基站高度（例如50m）
      f_c: 载波频率，默认为2 GHz
      sigma_dB: 阴影衰落标准差，单位dB，默认8 dB
    
    返回：
      信道增益（线性值）
    """
    c = 3e8  # 光速，单位 m/s
    lambda_ = c / f_c  # 波长
    # 构造3D位置（假设 UAV 高度为100m，基站高度为50m）
    user_3d = np.array([user_loc[0], user_loc[1], 100])
    bs_3d = np.array([bs_loc[0], bs_loc[1], 50])
    d = np.linalg.norm(user_3d - bs_3d)
    # 自由空间路径增益公式
    fspl = (lambda_ / (4 * np.pi * d)) ** 2
    # 加入对数正态阴影衰落
    shadowing_dB = np.random.normal(0, sigma_dB)
    shadowing = 10 ** (-shadowing_dB / 10)
    return fspl * shadowing


# 根据第二部分提供的计算函数，计算各 UAV 在单一信道上的速率与 SINR
def calculate_users_rates_per_channel(user_transmission_powers_one_channel,
                                      path_losses, user_bs_associations,
                                      max_user_rate=max_user_rate,
                                      consider_interference=True):
    num_users, num_bs = path_losses.shape
    # 每个 UAV 到所有 BS 的接收功率
    # 假设用户发射功率经过信道后在各 BS 的接收功率 = tx_power * path_loss
    users_at_bs = np.tile(user_transmission_powers_one_channel.reshape(-1, 1), (1, num_bs)) * path_losses
    # 各 BS 总接收功率
    bs_received = np.sum(users_at_bs, axis=0)
    # 每个 UAV 在其关联 BS 的接收功率
    selected_powers = np.array([users_at_bs[i, assoc] for i, assoc in enumerate(user_bs_associations)])
    # 干扰：除自身外，同 BS 的其他 UAV 发射造成的总接收功率
    interference = np.zeros(num_users)
    if consider_interference:
        for i in range(num_users):
            assoc = user_bs_associations[i]
            interference[i] = bs_received[assoc] - selected_powers[i]
    sinrs = selected_powers / (interference + NOISE)
    # 计算速率
    rates = np.log2(1 + sinrs)
    # 限制速率上限
    rates = np.clip(rates, 0, max_user_rate)
    return rates, sinrs


def calculate_tx_rate(tx_power_W, interference, channel_gain, noise, bandwidth_Hz):
    """
    根据 Shannon 定理计算传输速率 (Mbps)
    
    参数：
      tx_power_W: 发射功率（单位：W）
      interference: 干扰功率（单位：W）
      channel_gain: 信道增益（线性值）
      noise: 噪声功率（单位：W）
      bandwidth_Hz: 带宽（单位：Hz）
      
    返回：
      传输速率（单位：Mbps）
    """
    # 计算 SNR（线性值）
    SNR_linear = (tx_power_W * channel_gain) / (interference + noise)
    # 利用 Shannon 定理计算容量（单位：bps）
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    # 转换为 Mbps
    return capacity_bps / 1e6