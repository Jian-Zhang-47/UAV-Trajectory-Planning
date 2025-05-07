import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from config import *
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import tensorflow as tf

# GAT (Graph Attention Network) Model Definition -------------------------
class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATNet, self).__init__()
        # Define two GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        # Apply GAT layers and use ELU activation function
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        # Return the output after applying log softmax
        return F.log_softmax(x, dim=1)

# Function to assign channels to UAVs based on their positions -------------------
# def channel_assignment(uavs_pos, num_channels=NUM_CHANNELS, num_uavs=NUM_UAVS, 
#                        epochs=GAT_EPOCHS, lr=LEARNING_RATE_GAT, hidden_dim=HIDDEN_DIM_GAT):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     uav_features = torch.tensor(uavs_pos)
    
#     # Create adjacency matrix representing UAVs as a graph
#     adj_matrix = torch.ones(num_uavs, num_uavs) - torch.eye(num_uavs)
#     edge_index, _ = dense_to_sparse(adj_matrix)
    
#     # Create data object for graph neural network
#     data = Data(x=uav_features, edge_index=edge_index).to(device)
    
#     # Initialize GAT model, optimizer and loss function
#     model = GATNet(input_dim=uav_features.shape[1], hidden_dim=hidden_dim, output_dim=num_channels).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.NLLLoss()
    
#     # Define the target channels for each UAV
#     target_channels = torch.tensor([i % num_channels for i in range(num_uavs)], dtype=torch.long).to(device)
    
#     # Training loop
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         data.x = data.x.float()
#         out = model(data.x, data.edge_index)
#         loss = loss_fn(out, target_channels)  # Calculate loss
#         loss.backward()
#         optimizer.step()
    
#     # Evaluate the model to get final channel assignments
#     model.eval()
#     with torch.no_grad():
#         final_output = model(data.x, data.edge_index)
#         channels_assigned = final_output.argmax(dim=1).cpu().numpy()
    
#     return channels_assigned

def channel_assignment():
    c = []
    for i in range(NUM_UAVS):
        c.append(0)
    return c


# Function to calculate the channel gain between UAV and base station -----------
def calculate_channel_gain(uav_pos, bs_loc, f_c=FREQUENCY, sigma_dB=8):
    # Convert UAV and base station positions to 3D coordinates
    uav_3d = np.array([uav_pos[0], uav_pos[1], UAV_HEIGHT])
    bs_3d = np.array([bs_loc[0], bs_loc[1], BS_HEIGHT])
    
    # Calculate the Euclidean distance between UAV and base station
    d = np.linalg.norm(uav_3d - bs_3d)

    # Calculate path loss in dB (using a simplified model)
    path_loss_dB = 20 * np.log10(d) + 20 * np.log10(f_c) - 147.55 + sigma_dB
    # Convert path loss to channel gain
    channel_gain = 10 ** (-path_loss_dB / 10)
    return channel_gain

# Function to calculate channel gains and user-base station associations over time
def calculate_path_losses_and_associations_all_time(user_locations_all_time, bs_locations,
                                                    num_time_slots):
    num_users = user_locations_all_time.shape[1]
    num_base_stations = bs_locations.shape[0]

    channel_gain_all_time = np.zeros((num_time_slots, num_users, num_base_stations))

    # Loop through each time slot and calculate the channel gain for each user-base station pair
    for t in range(num_time_slots):
        for u in range(num_users):
            for b in range(num_base_stations):
                channel_gain_all_time[t, u, b] = calculate_channel_gain(user_locations_all_time[t, u, :],
                                                                        bs_locations[b])

    # Find the base station with the highest channel gain for each user
    user_bs_associations_num_all_time = channel_gain_all_time.argmax(axis=-1)

    return channel_gain_all_time, user_bs_associations_num_all_time

# Function to calculate the rate and SINR for each UAV on a single channel ------
def calculate_uav_rx_rates_per_channel(user_transmission_powers_one_channel,
                                      channel_gain, user_bs_associations, users_in_same_channel,
                                      max_user_rate = MAX_UAV_RATE,
                                      consider_interference=True):
    num_users, num_bs = channel_gain.shape
    
    # Calculate the received power at each base station
    uavs_at_bs = np.tile(user_transmission_powers_one_channel.reshape(-1, 1), (1, num_bs)) * channel_gain
    bs_received = np.sum(uavs_at_bs, axis=0)
    
    # Selected rx power for each uav based on its base station association
    selected_powers = np.array([uavs_at_bs[i, assoc] for i, assoc in enumerate(user_bs_associations)])
    interference = np.zeros(num_users)
    
    # Calculate interference from users transmitting on the same channel
    if consider_interference:
        for i in users_in_same_channel:
            assoc = user_bs_associations[i]
            interference[i] = bs_received[assoc] - selected_powers[i]
    
    # Calculate SINR (Signal to Interference + Noise Ratio)
    sinrs = selected_powers / (interference + NOISE_LEVEL)
    
    # Calculate the transmission rates based on the SINR values
    rates = np.log2(1 + sinrs)
    rates = np.clip(rates, 0, max_user_rate)  # Clip the rates to max_user_rate
    return rates, sinrs

# Function to calculate the transmission rate (Mbps) for each user -------------
def calculate_tx_rate(tx_power_W, interference, channel_gain, noise, bandwidth_Hz, efficiency=EFFICIENCY):
    # Calculate the SNR in linear scale
    SNR_linear = (tx_power_W * channel_gain) / (interference + noise)
    # Calculate the channel capacity (in bps) using Shannon's formula
    capacity_bps = bandwidth_Hz * np.log2(1 + SNR_linear)
    achievable_rate_bps = efficiency * capacity_bps
    achievable_rate_Mbps = achievable_rate_bps / 1e6
    return achievable_rate_Mbps  # Convert from bps to Mbps
