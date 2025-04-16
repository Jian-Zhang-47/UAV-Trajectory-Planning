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


# Assign base station to uav
def assign_base_stations(uav_positions, bs_locations):
    num_uavs = uav_positions.shape[0]
    num_bs = bs_locations.shape[0]
    path_losses = np.zeros((num_uavs, num_bs))
    for i in range(num_uavs):
        for j in range(num_bs):
            path_losses[i, j] = calculate_channel_gain(uav_positions[i], bs_locations[j])
    associations = np.argmax(path_losses, axis=1)
    return path_losses, associations

# Calculate channel gain using free space path loss combined with log-normal shadow fading model
def calculate_channel_gain(user_loc, bs_loc, f_c=FREQUENCY, sigma_dB=8):
    c = 3e8
    lambda_ = c / f_c
    user_3d = np.array([user_loc[0], user_loc[1], height_uav])
    bs_3d = np.array([bs_loc[0], bs_loc[1], height_bs])
    d = np.linalg.norm(user_3d - bs_3d)
    fspl = (lambda_ / (4 * np.pi * d)) ** 2
    shadowing_dB = np.random.normal(0, sigma_dB)
    shadowing = 10 ** (-shadowing_dB / 10)
    return fspl * shadowing


# Calculate the rate and SINR of each UAV on a single channel
def calculate_users_rates_per_channel(user_transmission_powers_one_channel,
                                      path_losses, user_bs_associations,
                                      max_user_rate=max_user_rate,
                                      consider_interference=True):
    num_users, num_bs = path_losses.shape
    users_at_bs = np.tile(user_transmission_powers_one_channel.reshape(-1, 1), (1, num_bs)) * path_losses
    bs_received = np.sum(users_at_bs, axis=0)
    selected_powers = np.array([users_at_bs[i, assoc] for i, assoc in enumerate(user_bs_associations)])
    interference = np.zeros(num_users)
    if consider_interference:
        for i in range(num_users):
            assoc = user_bs_associations[i]
            interference[i] = bs_received[assoc] - selected_powers[i]
    sinrs = selected_powers / (interference + NOISE)
    rates = np.log2(1 + sinrs)
    rates = np.clip(rates, 0, max_user_rate)
    return rates, sinrs

# Calculate transfer rate (Mbps)
def calculate_tx_rate(tx_power_W, interference, channel_gain, noise, bandwidth_Hz):
    SNR_linear = (tx_power_W * channel_gain) / (interference + noise)
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    return capacity_bps / 1e6


def calculate_path_loss_user_bs(user_loc, bs_loc, alpha_path_loss=2):
    # alpha_path_loss: path-loss exponent

    user_loc = np.concatenate([user_loc, [height_uav]])
    bs_loc = np.concatenate([bs_loc, [height_bs]])
    user_3d = np.array([user_loc[0], user_loc[1], height_uav])
    bs_3d = np.array([bs_loc[0], bs_loc[1], height_bs])
    d = np.linalg.norm(user_3d - bs_3d)
    path_loss =  d** (- alpha_path_loss)

    return path_loss

def calculate_path_losses_and_associations_all_time(user_locations_all_time, bs_locations,
                                                    num_time_slots):
    num_users = user_locations_all_time.shape[1]
    num_base_stations = bs_locations.shape[0]

    path_losses_all_time = np.zeros((num_time_slots, num_users, num_base_stations))

    for t in range(num_time_slots):
        for u in range(num_users):
            for b in range(num_base_stations):
                path_losses_all_time[t, u, b] = calculate_path_loss_user_bs(user_locations_all_time[t, u, :],
                                                                            bs_locations[b])

    user_bs_associations_num_all_time = path_losses_all_time.argmax(axis=-1)

    return path_losses_all_time, user_bs_associations_num_all_time