import gymnasium
import matplotlib.pyplot as plt
from gymnasium import spaces
import numpy as np
import itertools
from config import *
from environment import *
from path_planning import *
from communication import *

class Env(gymnasium.Env):
    def __init__(self, user_route, base_stations_locations, num_time_slots,
                 num_users=NUM_UAVS, num_base_stations=NUM_BSS,
                 num_channels=NUM_CHANNELS, velocity=UAV_SPEED):

        super(Env, self).__init__()

        # Initialize environment parameters
        self.user_route = user_route
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_time_slots = num_time_slots
        self.num_channels = num_channels
        self.velocity = velocity

        # Generate possible power level combinations for each channel
        self.power_level_all_channels = [x for x in itertools.product(TX_POWER_LEVELS, repeat=self.num_channels)
                                         if MIN_TX_POWER <= sum(x) <= MAX_TX_POWER]

        # Define the action space (discrete choices for each UAV's power level)
        self.action_space = {i: spaces.Discrete(len(self.power_level_all_channels)) for i in range(self.num_users)}

        # Define the observation space (features of each UAV, including path loss and location)
        self.num_features = (self.num_base_stations + 2)  # Path loss to each BS + UAV location
        self.observation_space = {i: spaces.Box(low=0, high=1,
                                                shape=(HISTORY_SIZE, self.num_features),
                                                dtype=np.float32) 
                                  for i in range(self.num_users)}

        # Initialize observation buffers for each UAV
        self.observations = {i: np.zeros((HISTORY_SIZE, self.num_features)) for i in range(self.num_users)}

        # Time step initialization
        self.t = 0
        self.base_stations_locations = base_stations_locations

        # Pre-allocate storage for path losses, associations, transmission powers, rates, and SINRs
        self.path_losses_all_time = np.zeros((self.num_time_slots, self.num_users, self.num_base_stations))
        self.user_bs_associations_num_all_time = np.zeros((self.num_time_slots, self.num_users), dtype=int)
        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.features_history = np.zeros((self.num_time_slots, self.num_users, self.num_features))

    def reset(self, **kwargs):

        self.t = 0
        episode_num = kwargs['episode_num']

        # Reset data for each time slot
        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))

        # Generate or reset user location data for the first episode
        if episode_num == 0:
            self.user_locations_all_time = determine_users_locations(self.user_route)
            self.path_losses_all_time, self.user_bs_associations_num_all_time = calculate_path_losses_and_associations_all_time(
                self.user_locations_all_time, self.base_stations_locations, self.num_time_slots
            )

        # Reset observations
        self.observations = {i: np.zeros((HISTORY_SIZE, self.num_features)) for i in range(self.num_users)}

        return self.observations, {}

    def step(self, action_dict, channel_assignment):

        # Reset transmission powers for the current time step
        self.user_transmission_powers_all_time[self.t, :, :] = 0

        # Assign transmission powers for each UAV based on the actions
        for i in range(self.num_users):
            power_vec = self.power_level_all_channels[action_dict[i]]
            channel = channel_assignment[i]
            self.user_transmission_powers_all_time[self.t, channel, i] = power_vec[channel]

        # Calculate rates and SINRs for each channel
        for c in range(self.num_channels):
            powers_c = self.user_transmission_powers_all_time[self.t, c, :]
            active_users = (powers_c > 0)

            if active_users.sum() == 0:
                self.rates_all_time[self.t, c, :] = 0
                self.sinrs_all_time[self.t, c, :] = 0
                continue

            rates_c, sinrs_c = calculate_users_rates_per_channel(
                powers_c,
                self.path_losses_all_time[self.t],
                self.user_bs_associations_num_all_time[self.t],
                users_in_same_channel=active_users
            )
            self.rates_all_time[self.t, c, :] = rates_c
            self.sinrs_all_time[self.t, c, :] = sinrs_c

        # Calculate the total rate for the users in this time step
        rate_per_user_this_time = self.rates_all_time[self.t].sum(axis=0)

        # Reward calculation: normalize rate by max UAV rate
        reward = {i: rate_per_user_this_time[i] / MAX_UAV_RATE for i in range(self.num_users)}

        # Determine if each UAV has reached its destination
        done = {i: (self.user_locations_all_time[self.t, i, :] == self.user_locations_all_time[self.t+1, i, :]).all() 
                for i in range(self.num_users)}

        # Move to the next time step
        self.t += 1

        # Prepare the UAV's next observation
        user_locations_normalized = self.user_locations_all_time[self.t]
        for i in range(self.num_users):
            path_loss_this_user_clipped = np.clip(self.path_losses_all_time[self.t, i, :].flatten(), 0, 10)

            # Update the UAV's history with the current observation
            self.observations[i][:-1, :] = self.observations[i][1:, :]
            self.observations[i][-1, :] = np.concatenate([path_loss_this_user_clipped, user_locations_normalized[i]])

            # Store the features for later use
            self.features_history[self.t, i] = self.observations[i][-1, :]

        return self.observations, reward, done, done, {}

