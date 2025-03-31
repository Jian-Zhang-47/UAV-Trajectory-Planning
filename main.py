# main.py
import numpy as np
from config import *
from environment import *
from target_assignment import merge_targets_dbscan, assign_targets_kmeans
from path_planning import optimize_route, route_cost_multi
from communication import channel_assignment
from simulation import *
from visualization import plot_env_trajectory
from ma_d3ql_model import*
import torch

# 1. Generate Environment
uavs = generate_uavs()
targets = generate_targets()
obstacles = generate_obstacles()
bss = generate_base_stations()

# 2. Merge Targets and Assign Clusters
clusters = merge_targets_dbscan(targets)
assigned_clusters = assign_targets_kmeans(uavs, clusters)
for i in range(NUM_UAVS):
    print(f"Number of clusters assigned to UAV {i+1}: {len(assigned_clusters[i])}")

# 3. Plan Path
optimal_paths = []
path_costs = []
for i in range(NUM_UAVS):
    if assigned_clusters[i].size > 0:
        waypoints = np.vstack((uavs[i], assigned_clusters[i], uavs[i]))
        best_route_idx = optimize_route(waypoints, obstacles, SAFE_RADIUS)
        best_route_points = [waypoints[idx] for idx in best_route_idx]
        cost_val = route_cost_multi(best_route_points, obstacles, SAFE_RADIUS)
        optimal_paths.append(np.array(best_route_points))
        path_costs.append(cost_val)
        print(f"Total cost of [UAV {i+1}] path: {cost_val:.2f}")
    else:
        optimal_paths.append(np.array([uavs[i]]))
        path_costs.append(0.0)
        print(f"Total cost of [UAV {i+1}] path: 0.00 (No target)")

# 4. Assign Channels
uav_features = torch.tensor(uavs, dtype=torch.float32)
channels_assigned, _ = channel_assignment(uav_features)
print("Channel assignment result:", channels_assigned)

# 6. 初始化 MA_D3QL 强化学习模型（用于传输功率决策）
# 状态维度：NUM_BASE_STATIONS + 2（基站信道增益 + UAV归一化位置）
num_features = NUM_BSS + 2
# 定义候选发射功率集合（离散动作）
power_level_all_channels = [0.01, 0.05, 0.1]
rl_agent = MA_D3QL(num_users=NUM_UAVS, num_channels=NUM_CHANNELS,
                    power_level_all_channels=power_level_all_channels,
                    num_features=num_features, algorithm='MA_D3QL')

# 5. UAV Energy Simulation
all_tx_points = []
for i in range(NUM_UAVS):
        print(f"\n--- UAV {i+1} 能量仿真（持续传输优化） ---")
        remaining_energy, flight_energy, tx_energy, tx_records, full_route, tx_points = simulate_uav_energy_continuous(
            i, optimal_paths[i], obstacles, SAFE_RADIUS, rl_agent, bss, channels_assigned
        )
        all_tx_points.append(tx_points)
        print(f"剩余能量: {remaining_energy:.2f}")
        print(f"飞行能耗: {flight_energy:.2f}, 传输能耗: {tx_energy:.2f}")
        for idx, (p, e, tx_duration, rate) in enumerate(tx_records):
            print(f"传输段 {idx+1}: RL选功率 = {p:.2f}W, 传输能耗 = {e:.2f}J, 时长 = {tx_duration:.2f}s, 速率 = {rate:.2f}Mbps")
        
# 6. Visualization
plot_env_trajectory(uavs, targets, clusters, obstacles, optimal_paths, path_costs, tx_points_list=all_tx_points)
