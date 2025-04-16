import numpy as np
import random
from config import *

# Determine whether the current point conflicts with obstacles
def is_segment_safe(current, candidate, obs, safe_distance, num_samples=10):
    for t in np.linspace(0, 1, num_samples):
        point = current + t * (candidate - current)
        if np.linalg.norm(point - obs) < safe_distance:
            return False
    return True

# Calculate the detour path from p1->p2 (considering obstacle safety distance)
def detour_edge_multi(p1, p2, obstacles, safe_distance, max_iter=20, init_epsilon=0.2):
    route = [p1]
    current = p1
    for _ in range(max_iter):
        line_vec = p2 - current
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-8:
            route.append(p2)
            return route

        violation_obs = None
        min_d = safe_distance
        for obs in obstacles:
            t = np.dot(obs - current, line_vec) / (line_len**2)
            t = np.clip(t, 0, 1)
            proj = current + t * line_vec
            d = np.linalg.norm(obs - proj)
            if d < min_d:
                min_d = d
                violation_obs = obs

        if violation_obs is None:
            route.append(p2)
            return route
        else:
            u = line_vec / line_len
            n1 = np.array([-u[1], u[0]])
            n2 = -n1

            epsilon = init_epsilon
            cand1 = violation_obs + n1 * safe_distance * (1 + epsilon)
            cand2 = violation_obs + n2 * safe_distance * (1 + epsilon)

            max_epsilon = 1.0
            while not is_segment_safe(current, cand1, violation_obs, safe_distance) and epsilon < max_epsilon:
                epsilon += 0.1
                cand1 = violation_obs + n1 * safe_distance * (1 + epsilon)
            epsilon = init_epsilon
            while not is_segment_safe(current, cand2, violation_obs, safe_distance) and epsilon < max_epsilon:
                epsilon += 0.1
                cand2 = violation_obs + n2 * safe_distance * (1 + epsilon)
            
            L1 = np.linalg.norm(current - cand1) + np.linalg.norm(cand1 - p2)
            L2 = np.linalg.norm(current - cand2) + np.linalg.norm(cand2 - p2)
            detour_point = cand1 if L1 < L2 else cand2

            route.append(detour_point)
            current = detour_point

    route.append(p2)
    return route

# Calculate the distance between two points after taking detours into account
def detour_distance_multi(p1, p2, obstacles, safe_distance):
    sub_route = detour_edge_multi(p1, p2, obstacles, safe_distance)
    dist = 0
    for i in range(len(sub_route) - 1):
        dist += np.linalg.norm(sub_route[i+1] - sub_route[i])
    return dist

# Calculate the total detour distance for the entire path
def route_cost_multi(route_points, obstacles, safe_distance):
    total = 0
    for i in range(len(route_points) - 1):
        total += detour_distance_multi(route_points[i], route_points[i+1], obstacles, safe_distance)
    return total

# Generate initial path using nearest neighbor algorithm
def nearest_neighbor_route(waypoints, obstacles, safe_distance):
    n = len(waypoints)
    if n <= 1:
        return list(range(n))
    visited = [False] * n
    visited[0] = True
    route = [0]
    current = 0
    for _ in range(n - 2):
        next_idx = None
        min_cost = float('inf')
        for j in range(1, n - 1):
            if not visited[j]:
                c = detour_distance_multi(waypoints[current], waypoints[j], obstacles, safe_distance)
                if c < min_cost:
                    min_cost = c
                    next_idx = j
        route.append(next_idx)
        visited[next_idx] = True
        current = next_idx
    route.append(n - 1)
    return route

# 2-opt local search optimization path
def two_opt(route, waypoints, obstacles, safe_distance, improvement_threshold=1e-6):
    best_route = route[:]
    best_cost = route_cost_multi([waypoints[i] for i in best_route], obstacles, safe_distance)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route)-2):
            for j in range(i+1, len(best_route)-1):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_cost = route_cost_multi([waypoints[idx] for idx in new_route], obstacles, safe_distance)
                if new_cost < best_cost - improvement_threshold:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best_route

# Generate initial population (including nearest neighbor path)
def create_population(waypoints, obstacles, safe_distance, pop_size):
    n = len(waypoints)
    population = []
    nn_route = nearest_neighbor_route(waypoints, obstacles, safe_distance)
    population.append(nn_route)
    indices = list(range(1, n-1))
    for _ in range(pop_size - 1):
        rand_route = [0] + random.sample(indices, len(indices)) + [n-1]
        population.append(rand_route)
    return population

# Evaluate the total cost of each path in the population
def evaluate_population(population, waypoints, obstacles, safe_distance):
    costs = []
    for route in population:
        costs.append(route_cost_multi([waypoints[i] for i in route], obstacles, safe_distance))
    return np.array(costs)

# Select elite
def selection(population, costs, elite_size):
    idx_sorted = np.argsort(costs)
    new_pop = [population[i] for i in idx_sorted[:elite_size]]
    return new_pop, idx_sorted

# Cross
def crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(1, n-3)
    end = random.randint(start+1, n-2)
    child = [None] * n
    child[0], child[-1] = parent1[0], parent1[-1]
    child[start:end] = parent1[start:end]
    fill_pos = [i for i in range(1, n-1) if i < start or i >= end]
    parent2_seq = [g for g in parent2[1:-1] if g not in child]
    for i, pos in enumerate(fill_pos):
        child[pos] = parent2_seq[i]
    return child

# Mutation
def mutate(route, mutation_rate):
    new_route = route[:]
    n = len(new_route)
    for i in range(1, n-1):
        if random.random() < mutation_rate:
            j = random.randint(1, n-2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# Genetic Algorithm for TSP Path Solving (Combining GA and 2-opt)
def genetic_algorithm_tsp(waypoints, obstacles, safe_distance, pop_size, generations, elite_size, mutation_rate):
    population = create_population(waypoints, obstacles, safe_distance, pop_size)
    for _ in range(generations):
        costs = evaluate_population(population, waypoints, obstacles, safe_distance)
        new_pop, idx_sorted = selection(population, costs, elite_size)
        while len(new_pop) < pop_size:
            indices = np.random.choice(idx_sorted[:len(population)//2], 2, replace=False)
            p1 = population[int(indices[0])]
            p2 = population[int(indices[1])]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop
    final_costs = evaluate_population(population, waypoints, obstacles, safe_distance)
    best_route = population[int(np.argmin(final_costs))]
    return best_route

# First use GA to solve, then use 2-opt to locally optimize the TSP path
def optimize_route(waypoints, obstacles, safe_distance, pop_size=POP_SIZE, generations=GENERATIONS, elite_size=ELITE_SIZE, mutation_rate=MUTATION_RATE):
    best_route_ga = genetic_algorithm_tsp(waypoints, obstacles, safe_distance, pop_size, generations, elite_size, mutation_rate)
    best_route = two_opt(best_route_ga, waypoints, obstacles, safe_distance)
    return best_route


def generate_points(route, velocity):
    seg_lengths = np.sqrt(np.sum(np.diff(route, axis=0)**2, axis=1))
    cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0)
    total_length = cum_lengths[-1]
    total_time = total_length / velocity
    T = int(np.ceil(total_time))
    positions = [] 
    times = np.arange(0, T + 1)

    for t in times:
        d = t * velocity
        seg_idx = np.searchsorted(cum_lengths, d, side='right') - 1
        
        if seg_idx >= len(route) - 1:
            pos = route[-1]
        else:
            segment_start = route[seg_idx]
            segment_end = route[seg_idx + 1]
            d_seg = d - cum_lengths[seg_idx]
            seg_length = seg_lengths[seg_idx]
            ratio = d_seg / seg_length if seg_length != 0 else 0
            pos = segment_start + ratio * (segment_end - segment_start)
        positions.append(pos)

    positions = np.array(positions)
    return positions




def determine_users_locations(optimization_paths, num_users=NUM_UAVS,
                              velocity=SPEED
                              ):
    max_time_slots, user_points = time_slot_max(optimization_paths)
    user_locations_all_time = np.zeros((max_time_slots, num_users, 2))
    
    for u, points in enumerate(user_points):
        len_points = points.shape[0]
        user_locations_all_time[:len_points, u, :] = points
        if len_points < max_time_slots:
            last_coordinate = points[-1, :]
            user_locations_all_time[len_points:, u, :] = np.tile(last_coordinate, (max_time_slots - len_points, 1))
    
    return user_locations_all_time

def time_slot_max(optimization_paths, num_users=NUM_UAVS, velocity=SPEED):
    user_points = []
    for u in range(num_users):
        points = generate_points(optimization_paths[u], velocity)
        user_points.append(points)
    
    max_time_slots = max(points.shape[0] for points in user_points)
    return max_time_slots, user_points