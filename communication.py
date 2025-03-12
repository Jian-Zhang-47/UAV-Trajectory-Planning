import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import *



# dynamiclly optimize tx power: P_tx >= SNR_min * (interference + noise) / channel_gain
def dynamic_optimize_tx_power(interference = 1e-9, channel_gain = CHANNEL_GAIN, SNR_min=SNR_MIN, noise=NOISE, TX_POWER_MAX=TX_POWER_MAX):
    required_power = SNR_min * (interference + noise) / channel_gain
    return min(required_power, TX_POWER_MAX)

def calculate_tx_rate(tx_power_W, interference = 1e-9, channel_gain = CHANNEL_GAIN, noise = NOISE, bandwidth_Hz = BANDWIDTH):
    """
    Shannon's formula
    """
    SNR_linear = (tx_power_W * channel_gain) / (interference + noise)
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    return capacity_bps / 1e6  # 转换为Mbps

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
