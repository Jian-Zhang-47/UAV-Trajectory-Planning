import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from config import COVERAGE_RADIUS, SAFE_RADIUS, NUM_UAVS
from path_planning import detour_edge_multi

def plot_env_trajectory(uavs, bss, targets, clusters, obstacles, optimal_paths, path_costs, tx_points_list=None):

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(targets[:,0], targets[:,1], marker='x', color='blue', label="Original Targets")
    axs[0].scatter(clusters[:,0], clusters[:,1], marker='o', color='green', label="Clusters")
    axs[0].scatter(uavs[:,0], uavs[:,1], marker='s', color='red', label="UAV Initial Positions")
    axs[0].scatter(bss[:,0], bss[:,1], marker='^', color='black', label="Base Stations")
    axs[0].scatter(obstacles[:,0], obstacles[:,1], marker='X', color='black', label="Obstacles")
    for idx, obs in enumerate(obstacles):
        lab = "Obstacle Coverage" if idx==0 else None
        circ = Circle((obs[0], obs[1]), SAFE_RADIUS, edgecolor='black', facecolor='none', linestyle='--', label=lab)
        axs[0].add_patch(circ)
    axs[0].set_title("UAV Initial Position, Target, Clusters & Obstacles")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].legend()

    colors = [plt.cm.tab10(i) for i in range(NUM_UAVS)]
    for i, path in enumerate(optimal_paths):
        key_nodes = path.tolist()
        key_nodes_tuples = [tuple(np.round(np.array(k), 3)) for k in key_nodes]
        
        expanded_route = []
        for j in range(len(key_nodes) - 1):
            sub_route = detour_edge_multi(np.array(key_nodes[j]), np.array(key_nodes[j+1]), obstacles, SAFE_RADIUS)
            if j == 0:
                expanded_route.extend(sub_route)
            else:
                expanded_route.extend(sub_route[1:])
        
        x_coords = [p[0] for p in expanded_route]
        y_coords = [p[1] for p in expanded_route]
        col = colors[i]
        axs[1].plot(x_coords, y_coords, marker='o', color=col, label=f"UAV {i+1}")
        axs[1].text(expanded_route[0][0], expanded_route[0][1], f"{path_costs[i]:.1f}", fontsize=10, color=col)

        if tx_points_list is not None and i < len(tx_points_list):
            for pt in tx_points_list[i]:
                tx_circ = Circle((pt[0], pt[1]), COVERAGE_RADIUS, color='orange', alpha=0.3)
                axs[1].add_patch(tx_circ)

    axs[1].scatter(targets[:, 0], targets[:, 1], c='blue', marker='x', label="Original Targets")
    axs[1].scatter(bss[:,0], bss[:,1], marker='^', color='black', label="Base Stations")
    axs[1].scatter(obstacles[:, 0], obstacles[:, 1], marker='X', color='black', label="Obstacles")
    for idx, obs in enumerate(obstacles):
        lab = "Obstacle Coverage Area" if idx == 0 else None
        circ = Circle((obs[0], obs[1]), SAFE_RADIUS, edgecolor='black', facecolor='none', linestyle='--', label=lab)
        axs[1].add_patch(circ)

    axs[1].set_title("Optimized UAV Trajectories, Key Nodes & Transmission Points")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
