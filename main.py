# main.py
import numpy as np
from config import *
from environment import *
from target_assignment import merge_targets_dbscan, assign_targets_kmeans
from path_planning import *
from communication import channel_assignment
from simulation import *
from visualization import plot_env_trajectory
from simulation_runner import *
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

max_time_slot,_ = time_slot_max(optimal_paths)

# 4. Assign Channels
uav_features = torch.tensor(uavs, dtype=torch.float32)
channels_assigned, _ = channel_assignment(uav_features)
print("Channel assignment result:", channels_assigned)

# 5. Initialize the MA_D3QL reinforcement learning model
simulation_runner = SimulationRunner(user_route=optimal_paths, num_users=NUM_UAVS, num_base_stations=NUM_BSS, num_time_slots=max_time_slot, num_channels=NUM_CHANNELS,
                                             base_stations_locations=bss,velocity=SPEED)
simulation_runner.run_one_episode()

P_opt = np.load(f'{results_folder}/user_transmission_powers_all_time_{num_episodes_test-1}.npy')
# 6. UAV Energy Simulation
all_tx_points = []
for i in range(NUM_UAVS):
        print(f"\n--- UAV {i+1} energy simulation ---")
        remaining_energy, flight_energy, tx_energy, tx_records, full_route, tx_points = simulate_uav_energy(
            i, optimal_paths[i], obstacles, SAFE_RADIUS, P_opt, bss, channels_assigned
        )
        all_tx_points.append(tx_points)
        print(f"Residual Energy: {remaining_energy:.2f}")
        print(f"Flight cost: {flight_energy:.2f}, Tx cost: {tx_energy:.2f}")
        for idx, (rl_p, p, e, tx_duration, rate) in enumerate(tx_records):
            print(f"Tx segment {idx+1}: RL power = {rl_p:.2f}W, Choose power = {p:.2f}W, Tx cost = {e:.2f}J, Tx duration = {tx_duration:.2f}s, Rate = {rate:.2f}Mbps")
        
# 7. Visualization
plot_env_trajectory(uavs, bss, targets, clusters, obstacles, optimal_paths, path_costs, tx_points_list=all_tx_points)
