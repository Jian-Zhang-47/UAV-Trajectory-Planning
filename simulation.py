import numpy as np
from path_planning import detour_edge_multi
from communication import dynamic_optimize_tx_power, calculate_tx_rate
from config import *

# Simulate the flight energy consumption and transmission energy consumption of UAV
def simulate_uav_energy(uav_index, key_nodes, obstacles, safe_distance,
                        initial_energy=INITIAL_ENERGY):
    """
    模拟 UAV 在整个飞行过程中能量消耗情况：
      - 飞行能耗：沿着完整展开的绕行路径按距离和飞行速度计算 (基于能耗模型)
      - 传输能耗：仅在关键目标点进行数据传输时消耗
    """

    energy = initial_energy
    flight_energy_total = 0
    tx_energy_total = 0
    transmissions = []

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

    # Flight Power Consumption Model
    p_total = P_HOVER + K * (SPEED ** 3) + P_PAYLOAD + P_COMM

    # 2. Calculate flight energy consumption
    for i in range(len(expanded_route) - 1):
        p1 = np.array(expanded_route[i])
        p2 = np.array(expanded_route[i + 1])
        distance = np.linalg.norm(p2 - p1)
        flight_time = distance / SPEED
        flight_energy = p_total * flight_time

        energy -= flight_energy
        flight_energy_total += flight_energy

        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during flight!")
            break

    # 3. Data transmission: is only performed at key nodes (except the starting point and the end point).
    for i in range(1, len(key_nodes) - 1):
        interference_val = np.random.uniform(0.1, 1.0)
        channel_gain_val = np.random.uniform(0.5, 1.5)
        tx_power = dynamic_optimize_tx_power(interference_val, channel_gain_val, SNR_MIN, NOISE, TX_POWER_MAX)
        tx_duration = IMAGE_SIZE / calculate_tx_rate(tx_power)
        tx_energy = tx_power * tx_duration
        hover_energy = P_HOVER * tx_duration
        flight_energy_total += hover_energy
        total_energy = tx_energy + hover_energy
        energy -= total_energy
        tx_energy_total += tx_energy
        transmissions.append((tx_power, tx_energy, total_energy, tx_duration))

        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during transmission at key node {i}!")
            break

    return energy, flight_energy_total, tx_energy_total, transmissions, expanded_route
