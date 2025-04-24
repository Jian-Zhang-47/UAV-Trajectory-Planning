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
    expanded_route = []
    for i in range(len(key_nodes) - 1):
        p1 = np.array(key_nodes[i])
        p2 = np.array(key_nodes[i+1])
        sub_route = detour_edge_multi(p1, p2, obstacles, safe_distance)
        if i == 0:
            expanded_route.extend(sub_route)
        else:
            expanded_route.extend(sub_route[1:])
    p_fly = fly_power()
    segment_times = []
    for i in range(len(expanded_route) - 1):
        p1 = np.array(expanded_route[i])
        p2 = np.array(expanded_route[i+1])
        d = np.linalg.norm(p2 - p1)
        dt = d / SPEED
        segment_times.append(dt)
        flight_energy = p_fly * dt
        energy -= flight_energy
        flight_energy_total += flight_energy
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during flight!")
            break
    total_flight_time = sum(segment_times)
    
    # Calculation of continuous transmission energy consumption
    num_tx_events = int(np.floor(total_flight_time / T_TRANSMISSION))
    
    cum_times = np.cumsum(segment_times)
    tx_points = []
    for event in range(num_tx_events):
        event_time = (event + 1) * T_TRANSMISSION
        seg_idx = np.searchsorted(cum_times, event_time)
        if seg_idx >= len(expanded_route)-1:
            pos = np.array(expanded_route[-1])
        else:
            t0 = cum_times[seg_idx-1] if seg_idx > 0 else 0
            t1 = cum_times[seg_idx]
            ratio = (event_time - t0) / (t1 - t0)
            pos0 = np.array(expanded_route[seg_idx])
            pos1 = np.array(expanded_route[seg_idx+1])
            pos = pos0 + ratio * (pos1 - pos0)

        tx_points.append(pos.tolist())

        state_bs = []
        for bs in bs_locations:
            gain = calculate_channel_gain(pos, bs)
            state_bs.append(gain)
        state_pos = [pos[0]/AREA_SIZE, pos[1]/AREA_SIZE]
        state_vector = np.array(state_bs + state_pos)  # shape: (NUM_BASE_STATIONS + 2,)
        
        state_dict = {}
        for j in range(NUM_UAVS):
            state_dict[j] = state_vector.copy()
        
        actions = P_opt_all_time[num_tx_events]
        assigned_channel = channels_assigned_all_time[num_tx_events][uav_index]
        d3ql_power = actions[assigned_channel][uav_index]
        chosen_power = d3ql_power

        interference_val = 0
        for j in range(NUM_UAVS):
            if j != uav_index and channels_assigned_all_time[num_tx_events][j] == assigned_channel:
                power_j = P_opt_all_time[num_tx_events][assigned_channel][j]
                gain_j = state_dict[j][assigned_channel]
                interference_val += power_j * gain_j

        channel_gain_val = state_bs[assigned_channel]
        
        rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                 channel_gain=channel_gain_val, noise=NOISE, bandwidth_Hz=BANDWIDTH)

        tx_duration = IMAGE_SIZE / rate if rate > 0 else T_TRANSMISSION
        if tx_duration > T_TRANSMISSION:
            required_rate = IMAGE_SIZE / T_TRANSMISSION
            exponent = required_rate * 1e6 / BANDWIDTH
            P_required = ((2**exponent) - 1) * (interference_val + NOISE) / channel_gain_val
            final_power = min(max(chosen_power, P_required), TX_POWER_MAX)
            chosen_power = final_power
            rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                    channel_gain=channel_gain_val, noise=NOISE, bandwidth_Hz=BANDWIDTH)
            tx_duration = IMAGE_SIZE / rate if rate > 0 else T_TRANSMISSION
            if tx_duration > T_TRANSMISSION:
                tx_duration = T_TRANSMISSION
        
        E_tx = chosen_power * tx_duration
        energy -= E_tx
        tx_energy_total += E_tx
        tx_records.append((d3ql_power, chosen_power, assigned_channel, E_tx, tx_duration, rate))
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during transmission event {event+1}!")
            break

    return energy, flight_energy_total, tx_energy_total, tx_records, expanded_route, tx_points



