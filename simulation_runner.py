import os

import numpy as np
import torch

from config import *
from env import Env
from ma_d3ql_model import MA_D3QL


def make_dir(folder_name, parent):
    if not (folder_name in os.listdir(parent)):
        os.makedirs(parent + '/' + folder_name)


class SimulationRunner:

    def __init__(self, user_route, num_users, num_base_stations, num_time_slots, num_channels, base_stations_locations,
                 velocity):
        self.user_route = user_route
        self.num_users = num_users
        self.num_base_stations = num_base_stations
        self.num_time_slots = num_time_slots
        self.num_channels = num_channels
        self.velocity = velocity
        self.base_stations_locations = base_stations_locations

        self.env = Env(num_time_slots=self.num_time_slots, user_route = self.user_route, base_stations_locations=self.base_stations_locations)

        self.saving_folder = results_folder

    def run_one_episode(self):
        if not os.path.exists(self.saving_folder):
            os.makedirs(self.saving_folder)
        print(f'starting ...')


        ma_d3ql = MA_D3QL(self.num_users, self.env.num_channels, self.env.power_level_all_channels, self.env.num_features, self.num_time_slots)

        ma_d3ql.run_training(self.env, self.num_time_slots, saving_folder=self.saving_folder)

        for ep in range(num_episodes_test):

            torch.manual_seed(seeds_test[ep])
            np.random.seed(seeds_test[ep])

            # Test environment
            # we set episode_num to zero for a new plane with each reset
            state, _ = self.env.reset(episode_num=0)
            done = {i: False for i in range(self.num_users)}

            while not any(done.values()):

                action = ma_d3ql.make_action_for_all_users(state, deterministic=True)

                state, reward, done, _, _ = self.env.step(action)


            np.save(f'{self.saving_folder}/user_locations_all_time_{ep}.npy',
                    self.env.user_locations_all_time)
            np.save(f'{self.saving_folder}/bs_locations_{ep}.npy',
                    self.env.base_stations_locations)
            np.save(f'{self.saving_folder}/user_bs_associations_num_all_time_{ep}.npy',
                    self.env.user_bs_associations_num_all_time)
            np.save(f'{self.saving_folder}/rates_all_time_{ep}.npy',
                    self.env.rates_all_time)
            np.save(f'{self.saving_folder}/user_transmission_powers_all_time_{ep}.npy',
                    self.env.user_transmission_powers_all_time)
            np.save(f'{self.saving_folder}/features_history_{ep}.npy',
                    self.env.features_history)

        self.env.close()