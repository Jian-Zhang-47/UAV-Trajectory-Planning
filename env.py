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
        self.quantization_level_all_times = np.zeros((self.num_time_slots, self.num_users))

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

    def step(self, action_dict):
        # Execute one time step within the environment

        # Segment Quality Calculations --------------------------------------------------------------------------------
        self.user_transmission_powers_all_time[self.t, :, :] = np.array([list(self.power_level_all_channels[a])
                                                                         for a in action_dict.values()]).T

        for c in range(self.num_channels):
            self.rates_all_time[self.t, c, :], self.sinrs_all_time[self.t, c, :] = \
                calculate_users_rates_per_channel(self.user_transmission_powers_all_time[self.t, c, :],
                                                  self.path_losses_all_time[self.t, :],
                                                  self.user_bs_associations_num_all_time[self.t, :],
                                                  consider_interference=True)

        rate_per_user_this_time = self.rates_all_time[self.t, :, :].sum(axis=0)


        # Reward Calculations -----------------------------------------------------------------------------------------
        reward = {i: rate_per_user_this_time.mean() / max_user_rate for i in range(self.num_users)}
        

        # Debugging ---------------------------------------------------------------------------------------------------
        if self.t in self.debug_time_slots:
            print('actions                      :\n', self.user_transmission_powers_all_time[self.t])
            print('rate per user                :', rate_per_user_this_time)
            print('reward                       :', list(reward.values()))

            fig, ax = plt.subplots(1)
            ax.set_title(f'Qualities at {self.t}')
            plt.show()

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