import numpy as np
from config import AREA_SIZE, NUM_UAVS, NUM_TARGETS, NUM_OBSTACLES

def generate_uavs():
    return np.random.uniform(0, AREA_SIZE, (NUM_UAVS, 2))

def generate_targets():
    return np.random.uniform(0, AREA_SIZE, (NUM_TARGETS, 2))

def generate_obstacles():
    return np.random.uniform(0, AREA_SIZE, (NUM_OBSTACLES, 2))
