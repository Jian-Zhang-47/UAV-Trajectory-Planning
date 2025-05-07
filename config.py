import numpy as np
import random
import torch

# Environment Configuration -------------------------------------------------
AREA_SIZE = 1000            # Area size (meters)
NUM_UAVS = 3               # Number of UAVs
NUM_BSS = 1                # Number of Base Stations (BSs)
NUM_TARGETS = 30           # Number of targets in the environment
NUM_OBSTACLES = 2          # Number of obstacles
COVERAGE_RADIUS = 50       # UAV coverage radius and target merging radius (meters)
SAFE_RADIUS = 50           # Safety distance from obstacles (meters)
UAV_HEIGHT = 100           # UAV flying height (meters)
BS_HEIGHT = 50             # Base station height (meters)

# UAV Specifications -------------------------------------------------------
INITIAL_ENERGY = 2.5e5     # UAV energy limit (Joules)
UAV_SPEED = 10.33              # UAV speed (m/s)
ROTOR_TIP_SPEED = 120      # Tip speed of rotor blade (m/s)
PARASITE_DRAG_COEFF = 0.6  # Parasite drag coefficient
AIR_DENSITY = 1.225        # Air density (kg/m^3)
ROTOR_SOLIDITY = 0.05      # Rotor solidity
ROTOR_DISK_AREA = 0.503    # Rotor disk area (m^2)
PROFILE_DRAG_COEFF = 0.012 # Profile drag coefficient
BLADE_ANGULAR_VELOCITY = 300  # Blade angular velocity (rad/s)
ROTOR_RADIUS = 0.4         # Rotor radius (m)
AIRCRAFT_WEIGHT = 20       # Aircraft weight (N)
CORRECTION_FACTOR = 0.1    # Incremental correction factor to induced power

# Transmission Configuration ----------------------------------------------
TRANSMISSION_TIME = COVERAGE_RADIUS / UAV_SPEED  # Transmission time (seconds)
MIN_SNR = 10               # Minimum Signal-to-Noise Ratio (SNR)
NOISE_LEVEL = 1e-13        # Noise level (W)
BANDWIDTH = 20e6           # Transmission bandwidth (Hz)
IMAGE_SIZE = 20            # Image size (MB)
MAX_UAV_RATE = 10          # Maximum UAV rate (Mbps)
FREQUENCY = 2.4e9          # Frequency (Hz)
NUM_CHANNELS = 1           # Number of channels available
MAX_TX_POWER = 0.1         # Maximum transmission power (Watts)
MIN_TX_POWER = 0.01        # Minimum transmission power (Watts)
TX_POWER_LEVELS = np.concatenate((np.array([MIN_TX_POWER]), np.linspace(0.05, MAX_TX_POWER, 20)))
EFFICIENCY = 0.9           # Optional efficiency factor for tx rate

# Genetic Algorithm (GA) Parameters ---------------------------------------
POPULATION_SIZE = 300      # Population size for GA
NUM_GENERATIONS = 1000     # Number of generations for GA
ELITE_SIZE = 5             # Number of elite individuals
MUTATION_RATE = 0.02       # Probability of mutation

# Graph Attention Network (GAT) Hyperparameters --------------------------
GAT_EPOCHS = 200
LEARNING_RATE_GAT = 0.005
HIDDEN_DIM_GAT = 8

# Random Seed for Reproducibility ----------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Simulation Configuration -------------------------------------------------
NUM_TRAIN_EPISODES = 1    # Number of training episodes
NUM_TEST_EPISODES = 1      # Number of test episodes
RESULTS_FOLDER = 'results' # Directory to save results
VERBOSE_MODE = 0           # Verbosity level for output
SAVE_MODEL_AFTER_TRAIN = True  # Flag to save the model after training
LOAD_PRETRAINED_MODEL = (NUM_TRAIN_EPISODES == 0)  # Flag to load pretrained model for testing
TEST_SEEDS = list(range(NUM_TEST_EPISODES))  # List of test seeds

# Learning Parameters -----------------------------------------------------
FC_LAYER_SIZES = [128, 64]      # Fully connected layer sizes for neural network
LSTM_STATE_SIZE = 256           # LSTM state size

# Exploration Strategy (Epsilon-Greedy) ----------------------------------
EPSILON_INITIAL = 1            # Initial epsilon for exploration
EPSILON_DECAY = 0.9995         # Decay factor for epsilon
EPSILON_MIN = 0.001            # Minimum epsilon value

BUFFER_CAPACITY = 1000         # Capacity of the experience replay buffer
BATCH_SIZE = 64                # Batch size for training
TARGET_UPDATE_INTERVAL = 20    # Interval to update target network
LEARNING_RATE = 0.001          # Learning rate for optimizer
HISTORY_SIZE = 4               # Size of the history window for learning
DISCOUNT_FACTOR = 0.75         # Discount factor (gamma) for Q-learning
