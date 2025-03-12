# main.py
import numpy as np
from config import *
from environment import generate_uavs, generate_targets, generate_obstacles
from target_assignment import merge_targets_dbscan, assign_targets_kmeans
from path_planning import optimize_route, detour_edge_multi, route_cost_multi
from communication import channel_assignment
from simulation import simulate_uav_energy
from visualization import plot_env_trajectory
import torch

# 1. Generate Environment
uavs = generate_uavs()
targets = generate_targets()
obstacles = generate_obstacles()

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

# 5. UAV Energy Simulation
for i in range(NUM_UAVS):
    print(f"\n--- UAV {i+1} Energy Simulation ---")
    expanded = []
    waypts = optimal_paths[i].tolist()
    for j in range(len(waypts)-1):
        sub_r = detour_edge_multi(np.array(waypts[j]), np.array(waypts[j+1]), obstacles, SAFE_RADIUS)
        if j == 0:
            expanded.extend(sub_r)
        else:
            expanded.extend(sub_r[1:])
    remaining_energy, flight_energy, tx_energy, tx_records, full_route = simulate_uav_energy(
        i, optimal_paths[i], obstacles, SAFE_RADIUS
    )
    print(f"Residual energy: {remaining_energy:.2f}")
    print(f"Flight energy cost: {flight_energy:.2f}, Transmission energy cost: {tx_energy:.2f}")
    for idx, (p, e, total_e, t) in enumerate(tx_records):
        print(f"Transmission No. {idx+1}: optimized power = {p:.2f}, tx energy cost = {e:.2f}, tx and hover energy cost = {total_e:.2f}, tx duration = {t:.2f}")

# 6. Visualization
plot_env_trajectory(uavs, targets, clusters, obstacles, optimal_paths, path_costs)
