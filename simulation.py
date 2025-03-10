import numpy as np
from path_planning import detour_edge_multi
from communication import dynamic_optimize_tx_power
from config import INITIAL_ENERGY, FLIGHT_COST_FACTOR, TX_DURATION

# Simulate the flight energy consumption and transmission energy consumption of UAV
def simulate_uav_energy(uav_index, key_nodes, obstacles, safe_distance,
                        initial_energy=INITIAL_ENERGY, flight_cost_factor=FLIGHT_COST_FACTOR, tx_duration=TX_DURATION,
                        SNR_MIN=10, NOISE=1e-3, TX_POWER_MAX=10):
    energy = initial_energy
    flight_energy_total = 0
    tx_energy_total = 0
    transmissions = []  # record (tx_power, tx_energy)
    
    # 1. Calculate the complete detour path
    expanded_route = []
    for i in range(len(key_nodes) - 1):
        p1 = key_nodes[i]
        p2 = key_nodes[i + 1]
        sub_route = detour_edge_multi(np.array(p1), np.array(p2), obstacles, safe_distance)
        if i == 0:
            expanded_route.extend(sub_route)
        else:
            expanded_route.extend(sub_route[1:])
    
    # 2. Calculate the flight energy consumption
    for i in range(len(expanded_route) - 1):
        p1 = np.array(expanded_route[i])
        p2 = np.array(expanded_route[i + 1])
        distance = np.linalg.norm(p2 - p1)
        flight_energy = flight_cost_factor * distance
        energy -= flight_energy
        flight_energy_total += flight_energy
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during flight!")
            break

    # 3. Calculate the transmission energy consumption
    for i in range(1, len(key_nodes) - 1):
        interference_val = np.random.uniform(0.1, 1.0)
        channel_gain_val = np.random.uniform(0.5, 1.5)
        tx_power = dynamic_optimize_tx_power(interference_val, channel_gain_val, SNR_min=SNR_MIN, noise=NOISE, TX_POWER_MAX=TX_POWER_MAX)
        tx_energy = tx_power * tx_duration
        energy -= tx_energy
        tx_energy_total += tx_energy
        transmissions.append((tx_power, tx_energy))
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during transmission at key node {i}!")
            break

    return energy, flight_energy_total, tx_energy_total, transmissions, expanded_route

# Return: remaining energy, flight energy consumption, transmission energy consumption, each transmission record, and the complete path of the expansion
