import numpy as np
from config import *
from environment import generate_uavs, generate_targets, generate_obstacles, generate_base_stations
from target_assignment import merge_targets_dbscan, assign_targets_kmeans
from path_planning import optimize_route, route_cost_multi, time_slot_max
from simulation import simulate_uav_energy
from visualization import plot_env_trajectory
from simulation_runner import SimulationRunner
from tqdm import tqdm

# 1. Generate the Environment: UAVs, Targets, Obstacles, and Base Stations
uavs = generate_uavs()  # Generate initial positions for UAVs
targets = generate_targets()  # Generate target positions
obstacles = generate_obstacles()  # Generate obstacle positions
bss = generate_base_stations()  # Generate base station locations

# 2. Merge Targets and Assign Clusters to UAVs
# Merge targets using DBSCAN and assign clusters to UAVs using KMeans
clusters = merge_targets_dbscan(targets)
assigned_clusters = assign_targets_kmeans(uavs, clusters)

# Display the number of clusters assigned to each UAV
for i in range(NUM_UAVS):
    print(f"Number of clusters assigned to UAV {i+1}: {len(assigned_clusters[i])}")

# 3. Plan Path for Each UAV
optimal_paths = []  # List to store optimal paths for each UAV
path_costs = []  # List to store path costs for each UAV

# For each UAV, calculate the optimal path based on assigned clusters
for i in range(NUM_UAVS):
    if assigned_clusters[i].size > 0:
        # Define waypoints for the path: start from UAV, visit assigned targets, return to UAV
        waypoints = np.vstack((uavs[i], assigned_clusters[i], uavs[i]))
        
        # Optimize the route to minimize cost, considering obstacles and safety radius
        best_route_idx = optimize_route(waypoints, obstacles, SAFE_RADIUS)
        best_route_points = [waypoints[idx] for idx in best_route_idx]
        
        # Calculate the cost of the optimized route
        cost_val = route_cost_multi(best_route_points, obstacles, SAFE_RADIUS)
        
        optimal_paths.append(np.array(best_route_points))
        path_costs.append(cost_val)
        print(f"Total cost of [UAV {i+1}] path: {cost_val:.2f}")
    else:
        optimal_paths.append(np.array([uavs[i]]))  # No targets, path is just the UAV position
        path_costs.append(0.0)  # No cost for this UAV
        print(f"Total cost of [UAV {i+1}] path: 0.00 (No target)")

# Calculate the maximum time slot needed for the simulation based on the paths
max_time_slot, _ = time_slot_max(optimal_paths)

# 4. Initialize the MA-D3QL Reinforcement Learning Model and Run Simulation
simulation_runner = SimulationRunner(
    user_route=optimal_paths,
    num_users=NUM_UAVS,
    num_base_stations=NUM_BSS,
    num_time_slots=max_time_slot,
    num_channels=NUM_CHANNELS,
    base_stations_locations=bss,
    velocity=UAV_SPEED
)

# Run one episode of the simulation
simulation_runner.run_one_episode()

# Load the results of the simulation
P_opt_all_time = np.load(f'{RESULTS_FOLDER}/user_transmission_powers_all_time_{NUM_TEST_EPISODES-1}.npy')
channels_assigned_all_time = np.load(f'{RESULTS_FOLDER}/user_channel_associations_num_all_time_{NUM_TEST_EPISODES-1}.npy')

# 5. UAV Energy Simulation
all_tx_points = []  # List to store transmission points for all UAVs

# Simulate energy consumption for each UAV
for i in tqdm(range(NUM_UAVS), desc="Simulating energy"):
    print(f"\n--- UAV {i+1} energy simulation ---")
    
    # Simulate energy consumption for the UAV
    remaining_energy, flight_energy, tx_energy, tx_records, full_route, tx_points = simulate_uav_energy(
        i, optimal_paths[i], obstacles, SAFE_RADIUS, P_opt_all_time, bss, channels_assigned_all_time
    )
    
    # Store transmission points for visualization
    all_tx_points.append(tx_points)
    
    # Print energy and cost details
    print(f"Residual Energy: {remaining_energy:.2f}")
    print(f"Flight cost: {flight_energy:.2f}, Tx cost: {tx_energy:.2f}")
    
    # Print details of each transmission segment
    for idx, (rl_p, p, c, e, tx_duration, rate) in enumerate(tx_records):
        print(f"Tx segment {idx+1}: RL power = {rl_p:.2f}W, Choose power = {p:.2f}W, Choose channel = {c}, "
              f"Tx cost = {e:.2f}J, Tx duration = {tx_duration:.2f}s, Rate = {rate:.2f}Mbps")

# 6. Visualization: Plot the UAVs' Trajectory, Target Locations, and Transmission Points
plot_env_trajectory(
    uavs, bss, targets, clusters, obstacles, optimal_paths, path_costs, tx_points_list=all_tx_points
)
