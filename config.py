import numpy as np
import random
import torch

# Environment
AREA_SIZE = 1000       # Area size
NUM_UAVS = 3           # Number of UAVs
NUM_BSS = 2            # Number of BSs
NUM_TARGETS = 50       # Number of targets
NUM_OBSTACLES = 0      # Number of obstacles
COVERAGE_RADIUS = 50   # UAV coverage radius & merging targets radius
SAFE_RADIUS = 80       # Obstacle safety distance
height_uav = 100
height_bs = 50
# UAV
INITIAL_ENERGY = 77 * 3600   # UAV energy limit
SPEED = 8              # m/s
P_HOVER = 180.0        # Hover power (W)
K = 0.2                # Air resistance coefficient
P_PAYLOAD = 10.0       # Payload power consumption (W)
P_COMM = 5.0           # Communication consumption (W)

# Transmission
T_TRANSMISSION = COVERAGE_RADIUS/SPEED
SNR_MIN = 10           # Minimum SNR
NOISE = 1e-13          # Noise
BANDWIDTH = 20e6       # Bandwidth (Hz)
IMAGE_SIZE = 20        # Image size (MB)
max_user_rate = 10
FREQUENCY = 2.4e9
NUM_CHANNELS = 2       # Number of channels
TX_POWER_MAX = 0.1
power_level = [0.01, 0.05, 0.1]
# GA
POP_SIZE = 100         # Population size
GENERATIONS = 500      # Iterative algebra
ELITE_SIZE = 2         # Number of elite
MUTATION_RATE = 0.001    # Mutation probability



# Random Seed
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


# Simulation ---------------------------------------------------------------
num_time_slots_default = 120

num_episodes_train = 200
num_episodes_test = 1

verbose_default = 0
save_model_after_train = True
load_pretrained_model = (num_episodes_train == 0)

seeds_test = list(range(num_episodes_test))

# Learning -----------------------------------------------------------------

fc_sizes = [128, 64]
lstm_state_size = 256

epsilon_init = 1
epsilon_decay = 0.9995
epsilon_min = 0.001

buffer_capacity = 1000
batch_size = 64
replace_target_interval = 20
learning_rate = 0.001
history_size = 4
gamma = 0.75