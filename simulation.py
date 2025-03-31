import numpy as np
from path_planning import detour_edge_multi
from communication import *
from config import *

# Simulate the flight energy consumption and transmission energy consumption of UAV
def simulate_uav_energy_continuous(uav_index, key_nodes, obstacles, safe_distance, rl_agent,
                                     bs_locations, channels_assigned, initial_energy=INITIAL_ENERGY):
    """
    模拟 UAV 飞行与持续传输过程：
      1. 飞行能耗：按照 detour 路径计算飞行能耗（与原代码一致）
      2. 持续传输：
         UAV 每隔 T_TRANSMISSION 秒传输一幅画面，
         对于每个传输事件，根据 UAV 当前位置构造状态（包括 UAV 到各基站的信道增益及归一化位置），
         利用 MA_D3QL 模型决策发射功率；
         根据 Shannon 公式计算传输速率，并计算该幅画面的传输时长 tx_duration = min(IMAGE_SIZE/rate, T_TRANSMISSION)；
         传输能耗计算为 chosen_power * tx_duration。
    """
    energy = initial_energy
    flight_energy_total = 0
    tx_energy_total = 0
    tx_records = []
    
    
    # --- 1. 飞行能耗计算 ---
    expanded_route = []
    for i in range(len(key_nodes) - 1):
        p1 = np.array(key_nodes[i])
        p2 = np.array(key_nodes[i+1])
        sub_route = detour_edge_multi(p1, p2, obstacles, safe_distance)
        if i == 0:
            expanded_route.extend(sub_route)
        else:
            expanded_route.extend(sub_route[1:])
    p_total = P_HOVER + K * (SPEED ** 3) + P_PAYLOAD + P_COMM
    segment_times = []
    for i in range(len(expanded_route) - 1):
        p1 = np.array(expanded_route[i])
        p2 = np.array(expanded_route[i+1])
        d = np.linalg.norm(p2 - p1)
        dt = d / SPEED
        segment_times.append(dt)
        flight_energy = p_total * dt
        energy -= flight_energy
        flight_energy_total += flight_energy
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during flight!")
            break
    total_flight_time = sum(segment_times)
    
    # --- 2. 持续传输能耗计算 ---
    # UAV 每隔 T_TRANSMISSION 秒传输一幅画面，直至飞行结束
    num_tx_events = int(np.floor(total_flight_time / T_TRANSMISSION))
    
    # 计算每个传输事件时 UAV 在飞行路线上对应的位置（采用线性插值）
    # 首先构造每个飞行段的累计时间
    cum_times = np.cumsum(segment_times)
    tx_points = []
    for event in range(num_tx_events):
        event_time = (event + 1) * T_TRANSMISSION  # 当前传输事件的时刻
        # 找到 UAV 当前所在的路段
        seg_idx = np.searchsorted(cum_times, event_time)
        if seg_idx >= len(expanded_route)-1:
            pos = np.array(expanded_route[-1])
        else:
            # 在 seg_idx 段内做线性插值
            t0 = cum_times[seg_idx-1] if seg_idx > 0 else 0
            t1 = cum_times[seg_idx]
            ratio = (event_time - t0) / (t1 - t0)
            pos0 = np.array(expanded_route[seg_idx])
            pos1 = np.array(expanded_route[seg_idx+1])
            pos = pos0 + ratio * (pos1 - pos0)

        tx_points.append(pos.tolist())

        # 构造状态向量：前 NUM_BASE_STATIONS 个元素为 UAV 到各 BS 的信道增益，
        # 后 2 个元素为 UAV 位置归一化 (x/AREA_SIZE, y/AREA_SIZE)
        state_bs = []
        for bs in bs_locations:
            gain = calculate_channel_gain(pos, bs)
            state_bs.append(gain)
        state_pos = [pos[0]/AREA_SIZE, pos[1]/AREA_SIZE]
        state_vector = np.array(state_bs + state_pos)  # shape: (NUM_BASE_STATIONS + 2,)
        
        # 为避免 MA_D3QL 访问不到其他 UAV 状态，构造全 UAV 状态字典
        state_dict = {}
        for j in range(NUM_UAVS):
            # 此处可以使用当前 UAV 状态作为其他 UAV 的占位状态
            state_dict[j] = state_vector.copy()
        
        actions = rl_agent.make_action_for_all_users(state_dict, deterministic=True)
        action = actions[uav_index]
        chosen_power = rl_agent.power_level_all_channels[action]
        assigned_channel = channels_assigned[uav_index]
        
        # 模拟同一信道下其他 UAV 干扰（此处采用随机模拟）
        # 得到所有 UAV 的决策动作
        actions = rl_agent.make_action_for_all_users(state_dict, deterministic=True)

        # 当前 UAV 的决策及发射功率
        action = actions[uav_index]
        chosen_power = rl_agent.power_level_all_channels[action]
        assigned_channel = channels_assigned[uav_index]

        # 计算同一信道下其他 UAV 的干扰（真实计算）
        interference_val = 0
        for j in range(NUM_UAVS):
            if j != uav_index and channels_assigned[j] == assigned_channel:
                power_j = rl_agent.power_level_all_channels[actions[j]]
                # 从 state_dict 中提取 UAV j 到各基站的信道增益，
                # 假设 state_vector 的前 NUM_BASE_STATIONS 个元素分别为各基站的信道增益
                gain_j = state_dict[j][assigned_channel]
                interference_val += power_j * gain_j
        # UAV 到所属基站的信道增益为 state_bs[assigned_channel]
        channel_gain_val = state_bs[assigned_channel]
        
        # 计算传输速率（Mbps）——注意：速率可能为零
        rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                 channel_gain=channel_gain_val, noise=NOISE, bandwidth_Hz=BANDWIDTH)

        tx_duration = IMAGE_SIZE / rate if rate > 0 else T_TRANSMISSION
        if tx_duration > T_TRANSMISSION:
            # 要求速率至少满足：IMAGE_SIZE / rate_req <= T_TRANSMISSION，即 rate_req >= IMAGE_SIZE / T_TRANSMISSION
            required_rate = IMAGE_SIZE / T_TRANSMISSION  # 单位 Mbps
            # 根据 Shannon 公式求解最低发射功率 P_required，使得：
            # bandwidth_Hz * log2(1 + (P_required * channel_gain_val)/(interference_val+NOISE)) / 1e6 >= required_rate
            # 即 1 + (P_required * channel_gain_val)/(interference_val+NOISE) >= 2^(required_rate*1e6/bandwidth_Hz)
            exponent = required_rate * 1e6 / BANDWIDTH
            P_required = ((2**exponent) - 1) * (interference_val + NOISE) / channel_gain_val
            # 更新发射功率：选择当前决策和所需功率中的较大值，但不超过 TX_POWER_MAX
            final_power = min(max(chosen_power, P_required), TX_POWER_MAX)
            chosen_power = final_power
            # 重新计算速率和传输时长
            rate = calculate_tx_rate(chosen_power, interference=interference_val,
                                    channel_gain=channel_gain_val, noise=NOISE, bandwidth_Hz=BANDWIDTH)
            tx_duration = IMAGE_SIZE / rate if rate > 0 else T_TRANSMISSION
            # 如果此时仍超过 T_TRANSMISSION（比如已经达到 TX_POWER_MAX），就取 T_TRANSMISSION
            if tx_duration > T_TRANSMISSION:
                tx_duration = T_TRANSMISSION
        
        E_tx = chosen_power * tx_duration
        energy -= E_tx
        tx_energy_total += E_tx
        tx_records.append((chosen_power, E_tx, tx_duration, rate))
        
        if energy < 0:
            print(f"UAV {uav_index+1} energy depleted during transmission event {event+1}!")
            break

    return energy, flight_energy_total, tx_energy_total, tx_records, expanded_route, tx_points



