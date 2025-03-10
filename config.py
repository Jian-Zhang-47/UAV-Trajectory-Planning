import numpy as np
import random
import torch

# Environment
AREA_SIZE = 1000       # Area size
NUM_UAVS = 3           # Number of UAVs
NUM_TARGETS = 50       # Number of targets
NUM_OBSTACLES = 5      # Number of obstacles
COVERAGE_RADIUS = 50   # UAV coverage radius & merging targets radius
SAFE_RADIUS = 30       # Obstacle safety distance

# Energy
INITIAL_ENERGY = 100   # UAV energy limit
FLIGHT_COST_FACTOR = 0.01  # The factor of energy consumption per unit distance
TX_DURATION = 0.01  # Duration of data transfer

# Transmission
TX_POWER_MAX = 10      # Maximum transmission power
SNR_MIN = 10           # Minimum SNR
NOISE = 1e-3           # Noise

# GA
POP_SIZE = 100         # Population size
GENERATIONS = 500      # Iterative algebra
ELITE_SIZE = 2         # Number of elite
MUTATION_RATE = 0.1    # Mutation probability

# Channel
NUM_CHANNELS = 5       # Number of channels

# Random Seed
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)