import gymnasium
import matplotlib.pyplot as plt
from gymnasium import spaces
import numpy as np
from config import *
from environment import *
from path_planning import *
from communication import *
import itertools
class Env(gymnasium.Env):
    def __init__(self, user_route, base_stations_locations, num_time_slots,
                 num_users=NUM_UAVS, num_base_stations=NUM_BSS,
                 num_channels=NUM_CHANNELS,
                 velocity=SPEED,
                 debug_time_slots=()):  # 10, 60, 110
        super(Env, self).__init__()

        self.user_route = user_route
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_time_slots = num_time_slots
        self.num_channels = num_channels
        self.velocity = velocity
        self.debug_time_slots = debug_time_slots


        # actions: determine power level
        self.power_level_all_channels = [x for x in itertools.product(power_levels, repeat=self.num_channels)
                                         if TX_POWER_MIN <= sum(x) <= TX_POWER_MAX]

        self.action_space = {i: spaces.Discrete(len(self.power_level_all_channels)) for i in range(self.num_users)}

        self.num_features = (self.num_base_stations  # path loss to each BS
                                + 2  # UAV location
                                )

        self.observation_space = {i: spaces.Box(low=0, high=1,
                                                shape=(history_size, self.num_features),
                                                dtype=np.float32)
                                  for i in range(self.num_users)}

        self.observations = {i: np.zeros((history_size, self.num_features)) for i in range(self.num_users)}

        self.t = 0
        self.base_stations_locations = base_stations_locations

        self.path_losses_all_time = np.zeros((self.num_time_slots, self.num_users, self.num_base_stations))
        self.user_bs_associations_num_all_time = np.zeros((self.num_time_slots, self.num_users), dtype=int)

        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))

        self.features_history = np.zeros((self.num_time_slots, self.num_users, self.num_features))

    def reset(self, **kwargs):
        self.t = 0
        episode_num = kwargs['episode_num']

        self.user_transmission_powers_all_time = np.zeros((self.num_time_slots, self.num_channels,
                                                           self.num_users))
        self.rates_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))
        self.sinrs_all_time = np.zeros((self.num_time_slots, self.num_channels, self.num_users))


        if episode_num == 0:
            self.user_locations_all_time = determine_users_locations(self.user_route)
            self.path_losses_all_time, self.user_bs_associations_num_all_time \
                = calculate_path_losses_and_associations_all_time(self.user_locations_all_time,
                                                                  self.base_stations_locations,
                                                                  self.num_time_slots)

        # Reset the state of the environment to an initial state
        self.observations = {i: np.zeros((history_size, self.num_features)) for i in range(self.num_users)}

        return self.observations, {}

    def step(self, action_dict, channel_assignment):
        # Execute one time step within the environment
        self.user_transmission_powers_all_time[self.t, :, :] = 0
        for i in range(self.num_users):
            power_vec = self.power_level_all_channels[action_dict[i]]
            channel = channel_assignment[i]
            self.user_transmission_powers_all_time[self.t, channel, i] = power_vec[channel]

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

        rate_per_user_this_time = self.rates_all_time[self.t].sum(axis=0)


        # Reward Calculations -----------------------------------------------------------------------------------------
        reward = {i: rate_per_user_this_time[i] / max_user_rate for i in range(self.num_users)}

        # Next State Calculations -------------------------------------------------------------------------------------

        done = {i: (self.user_locations_all_time[self.t,i,:] == self.user_locations_all_time[self.t+1,i,:]).all() for i in range(self.num_users)}

        self.t += 1

        user_locations_normalized = self.user_locations_all_time[self.t]
        # making local observations
        for i in range(self.num_users):

            path_loss_this_user_clipped = np.clip(self.path_losses_all_time[self.t, i, :].flatten(), 0, 10)

            self.observations[i][:-1, :] = self.observations[i][1:, :]


            self.observations[i][-1, :] = np.concatenate([path_loss_this_user_clipped,
                                                              user_locations_normalized[i]])
            self.features_history[self.t, i] = self.observations[i][-1, :]

        return self.observations, reward, done, done, {}