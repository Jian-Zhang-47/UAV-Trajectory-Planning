import numpy as np
from path_planning import detour_edge_multi
from communication import *
from config import *
from fly_power import *

# Simulate the flight energy consumption and transmission energy consumption of UAV
def simulate_uav_energy(uav_index, key_nodes, obstacles, safe_distance, P_opt_all_time,
                        bs_locations, channels_assigned_all_time, initial_energy=INITIAL_ENERGY):
    energy = initial_energy
    flight_energy_total = 0
    tx_energy_total = 0
    tx_records = []

    # Flight energy consumption calculation
    expanded_route = []  # Full route with detours
    for i in range(len(key_nodes) - 1):
        p1 = np.array(key_nodes[i])
        p2 = np.array(key_nodes[i+1])
        
        # Add detours to the route to avoid obstacles
        sub_route = detour_edge_multi(p1, p2, obstacles, safe_distance)
        expanded_route.extend(sub_route if i == 0 else sub_route[1:])  # Avoid duplicate points

    p_fly = fly_power()  # Power consumption during flight
    segment_times = []  # Time durations for each flight segment

    # Calculate energy consumption for each segment
    for i in range(len(expanded_route) - 1):
        p1 = np.array(expanded_route[i])
        p2 = np.array(expanded_route[i+1])
        distance = np.linalg.norm(p2 - p1)
        segment_time = distance / UAV_SPEED  # Time to traverse the segment
        segment_times.append(segment_time)
        
        flight_energy = p_fly * segment_time  # Energy consumed in this segment
        energy -= flight_energy
        flight_energy_total += flight_energy
        
        if energy < 0:  # Check if energy is depleted during flight
            print(f"UAV {uav_index+1} energy depleted during flight!")
            break

    total_flight_time = sum(segment_times)

    # Transmission energy consumption calculation
    num_tx_events = int(np.floor(total_flight_time / TRANSMISSION_TIME))  # Number of transmission events

    # Determine transmission points based on the flight route
    cum_times = np.cumsum(segment_times)
    tx_points = []
    for event in range(num_tx_events):
        event_time = (event + 1) * TRANSMISSION_TIME
        seg_idx = np.searchsorted(cum_times, event_time)

        # Calculate the position of the UAV at the transmission event
        if seg_idx >= len(expanded_route) - 1:
            pos = np.array(expanded_route[-1])
        else:
            t0 = cum_times[seg_idx - 1] if seg_idx > 0 else 0
            t1 = cum_times[seg_idx]
            ratio = (event_time - t0) / (t1 - t0)
            pos0 = np.array(expanded_route[seg_idx])
            pos1 = np.array(expanded_route[seg_idx + 1])
            pos = pos0 + ratio * (pos1 - pos0)

        tx_points.append(pos.tolist())  # Append the transmission position

        # Calculate the channel gain for each base station
        state_bs = [calculate_channel_gain(pos, bs) for bs in bs_locations]
        state_pos = [pos[0] / AREA_SIZE, pos[1] / AREA_SIZE]
        state_vector = np.array(state_bs + state_pos)  # Concatenate base station gains and position
        
        # State dictionary for each UAV (same state vector for all UAVs)
        state_dict = {j: state_vector.copy() for j in range(NUM_UAVS)}

        # Get transmission power from optimization actions and assigned channel
        actions = P_opt_all_time[event]
        assigned_channel = channels_assigned_all_time[event][uav_index]
        chosen_power = actions[assigned_channel][uav_index]

        # Calculate interference from other UAVs
        interference_val = 0
        for j in range(NUM_UAVS):
            if j != uav_index and channels_assigned_all_time[event][j] == assigned_channel:
                power_j = P_opt_all_time[event][assigned_channel][j]
                gain_j = state_dict[j][assigned_channel]
                interference_val += power_j * gain_j

        # Calculate the transmission rate based on the chosen power and interference
        channel_gain_val = state_bs[assigned_channel]
        rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                 channel_gain=channel_gain_val, noise=NOISE_LEVEL, bandwidth_Hz=BANDWIDTH)
        # Ensure transmission rate is sufficient to transmit the image within the allowed time
        rate = float(np.array(rate))

        tx_duration = IMAGE_SIZE / rate if rate > 0 else TRANSMISSION_TIME
        if tx_duration > TRANSMISSION_TIME:
            required_rate = IMAGE_SIZE / TRANSMISSION_TIME
            exponent = required_rate * 1e6 / BANDWIDTH
            P_required = ((2**exponent) - 1) * (interference_val + NOISE_LEVEL) / channel_gain_val
            chosen_power = min(max(chosen_power, P_required), MAX_TX_POWER)
            rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                    channel_gain=channel_gain_val, noise=NOISE_LEVEL, bandwidth_Hz=BANDWIDTH)
            tx_duration = IMAGE_SIZE / rate if rate > 0 else TRANSMISSION_TIME
            if tx_duration > TRANSMISSION_TIME:
                tx_duration = TRANSMISSION_TIME
        
        # Calculate transmission energy and update UAV's energy
        E_tx = chosen_power * tx_duration
        energy -= E_tx
        tx_energy_total += E_tx
        tx_records.append((chosen_power, chosen_power, assigned_channel, E_tx, tx_duration, rate))

        if energy < 0:  # Check if energy is depleted during transmission
            print(f"UAV {uav_index+1} energy depleted during transmission event {event + 1}!")
            break

    return energy, flight_energy_total, tx_energy_total, tx_records, expanded_route, tx_points
