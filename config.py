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

# UAV
INITIAL_ENERGY = 77 * 3600   # UAV energy limit
SPEED = 8              # m/s
P_HOVER = 180.0        # Hover power (W)
K = 0.2                # Air resistance coefficient
P_PAYLOAD = 10.0       # Payload power consumption (W)
P_COMM = 5.0           # Communication consumption (W)

# Transmission
TX_POWER_MAX = 0.1     # Maximum transmission power (W)
SNR_MIN = 10           # Minimum SNR
NOISE = 1e-13          # Noise
CHANNEL_GAIN = 1e-7    # Channel gain
BANDWIDTH = 20e6       # Bandwidth (Hz)
IMAGE_SIZE = 10        # Image size (MB)

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