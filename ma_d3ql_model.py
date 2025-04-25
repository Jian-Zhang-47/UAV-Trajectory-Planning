from copy import deepcopy
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

from config import *
from model import *
from communication import channel_assignment


def get_numpy_from_dict_values(x):
    return np.array(list(x.values()))


class MA_D3QL:
    def __init__(self, num_users, num_channels, power_level_all_channels, num_features, num_time_slots, model_name='ma_d3ql'):
        self.num_users = num_users
        self.num_channels = num_channels
        self.power_level_all_channels = power_level_all_channels
        self.model_name = model_name
        self.num_features = num_features
        self.num_time_slots = num_time_slots

        # Choose the device (GPU or CPU)
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'

        # Read configuration from config file
        self.load_pretrained_model = LOAD_PRETRAINED_MODEL
        self.batch_size = BATCH_SIZE
        self.replace_target_interval = TARGET_UPDATE_INTERVAL
        self.gamma = DISCOUNT_FACTOR

        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.num_users, self.num_features, self.device)

        # Define loss function (MSE loss)
        self.loss = nn.MSELoss()

        # Epsilon for exploration-exploitation tradeoff
        self.epsilon = EPSILON_INITIAL

        # Create models and target models for each user
        self.models = np.empty(self.num_users, dtype=object)
        self.target_models = np.empty(self.num_users, dtype=object)
        self.models_initial_weights = np.empty(self.num_users, dtype=object)

        for i in range(self.num_users):
            self.models[i] = DeepQNetwork(name=f'{self.model_name}_model_{i}',
                                          num_features=num_features, num_actions=len(power_level_all_channels),
                                          device=self.device)
            self.target_models[i] = DeepQNetwork(name=f'{self.model_name}_target_model_{i}',
                                                 num_features=num_features, num_actions=len(power_level_all_channels),
                                                 device=self.device)

            # Load pretrained models if needed
            if self.load_pretrained_model:
                file = f'results/model_{i}.pt'
                self.models[i].load_checkpoint(file)

            # Copy initial weights to target models
            self.models_initial_weights[i] = self.models[i].state_dict()
            self.target_models[i].load_state_dict(self.models_initial_weights[i])

        # Initialize step counter for learning
        self.learn_step_counter = 0

        # Indexes for batch sampling
        self.indexes = np.arange(self.batch_size)

        # Initialize progress bar for training episodes
        self.T = tqdm(NUM_TRAIN_EPISODES, desc='Progress', leave=True, disable=bool(1 - VERBOSE_MODE))

    def run_training(self, env, num_time_slots, saving_folder):
        rewards_history = np.zeros((NUM_TRAIN_EPISODES, num_time_slots))
        loss_history = np.zeros((NUM_TRAIN_EPISODES, num_time_slots))
        epsilon_history = np.zeros((NUM_TRAIN_EPISODES, num_time_slots))

        self.T = tqdm(range(NUM_TRAIN_EPISODES), desc="Training episodes", unit="ep", leave=True)

        for ep in self.T:
            state, _ = env.reset(episode_num=ep, plane_this_episode=None)
            done = {i: False for i in range(self.num_users)}
            total_reward = 0

            with tqdm(total=num_time_slots, desc=f"Ep {ep+1}", leave=False, unit="ts") as pbar_ts:
                while not any(done.values()):
                    self.__update_epsilon()

                    prev_state = deepcopy(state)

                    action = self.make_action_for_all_users(state)
                    t = env.t
                    user_locations_now = env.user_locations_all_time[t]
                    channel_assignment_list = channel_assignment(user_locations_now)
                    channel_assignment_dict = {i: int(channel_assignment_list[i]) for i in range(self.num_users)}
                    state, reward, done, _, _ = env.step(action, channel_assignment_dict)
                    avg_reward = np.array(list(reward.values())).mean()
                    total_reward += avg_reward

                    # Store experience in the buffer
                    self.add_aggregated_experience_to_buffers(prev_state, state, action, reward, done)

                    # Train the model using random samples
                    loss = self.train_on_random_samples()

                    # Record history for later analysis
                    rewards_history[ep, env.t - 1] = avg_reward
                    loss_history[ep, env.t - 1] = loss
                    epsilon_history[ep, env.t - 1] = self.epsilon

                    pbar_ts.update(1)

            self.T.set_description(f"Reward: {(np.round(total_reward, 2))}")
            self.T.refresh()

        # Save training history and models
        np.save(f'{saving_folder}/all_rewards.npy', rewards_history)
        np.save(f'{saving_folder}/all_loss_values.npy', loss_history)
        np.save(f'{saving_folder}/all_epsilon_values.npy', epsilon_history)

        if SAVE_MODEL_AFTER_TRAIN:
            for i in range(self.num_users):
                self.models[i].save_checkpoint(f'./{saving_folder}/model_{i}')

    def make_action_for_all_users(self, state, deterministic=False):
        actions = {}

        for i in range(self.num_users):
            if (np.random.random() < self.epsilon) and (not deterministic):
                actions[i] = np.random.randint(len(self.power_level_all_channels))
            else:
                observation = th.tensor(state[i], dtype=th.float).to(self.device).unsqueeze(0)
                _, advantages = self.models[i].forward(observation)

                actions[i] = th.argmax(advantages).item()

        return actions

    def add_aggregated_experience_to_buffers(self, previous_observations, new_observations, actions, rewards, dones):
        previous_observations_arr = get_numpy_from_dict_values(previous_observations)
        new_observations_arr = get_numpy_from_dict_values(new_observations)
        actions_arr = get_numpy_from_dict_values(actions)
        rewards_arr = get_numpy_from_dict_values(rewards)
        dones_arr = get_numpy_from_dict_values(dones)

        self.buffer.store_experience(
            previous_observations_arr, new_observations_arr,
            actions_arr, rewards_arr, dones_arr)

    def __update_epsilon(self, reset=False):
        if not reset:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(self.epsilon, EPSILON_MIN)
        else:
            self.epsilon = EPSILON_INITIAL

    def __replace_target_networks(self, model_index):
        if self.learn_step_counter == 0 or (self.learn_step_counter % self.replace_target_interval) == 0:
            self.target_models[model_index].load_state_dict(self.models[model_index].state_dict())

    @staticmethod
    def __convert_value_advantage_to_q_values(v, a):
        return th.add(v, (a - a.mean(dim=1, keepdim=True)))

    def train_on_random_samples(self):
        if self.buffer.mem_counter < self.batch_size:
            return

        states, next_states, actions, reward, dones = self.buffer.sample_buffer()

        q_predicted_list = []
        q_next_list = []

        for i in range(self.num_users):
            self.models[i].train()
            self.models[i].optimizer.zero_grad()
            self.__replace_target_networks(model_index=i)

            V_states, A_states = self.models[i].forward(states[:, i, :])
            q_pred_agent = self.__convert_value_advantage_to_q_values(V_states, A_states)[self.indexes, actions[:, i]]
            q_predicted_list.append(q_pred_agent)

            with th.no_grad():
                _, A_next_states = self.models[i].forward(next_states[:, i, :])
                actions_next_states_best = A_next_states.argmax(axis=1).detach()

                V_next_states, A_next_states = self.target_models[i].forward(next_states[:, i, :])
                q_next_all_actions = self.__convert_value_advantage_to_q_values(V_next_states, A_next_states)
                q_next_agent = q_next_all_actions.gather(1, actions_next_states_best.unsqueeze(1)).squeeze()
                q_next_agent[dones[:, i]] = 0.0
                q_next_list.append(q_next_agent)

        q_predicted = th.stack(q_predicted_list, dim=1)
        q_next = th.stack(q_next_list, dim=1)

        q_target = th.nan_to_num(reward).mean(axis=-1).unsqueeze(-1) + (self.gamma * q_next)

        loss = self.loss(q_predicted, q_target).to(self.device)
        loss.backward()

        for i in range(self.num_users):
            self.models[i].optimizer.step()
            self.models[i].eval()

        self.learn_step_counter += 1

        return loss.detach().cpu().numpy()

    def get_weights(self):
        return self.model.state_dict(), self.target_model.state_dict()

    def set_weights(self, weights, weights_target):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights_target)

        self.model.lstm.flatten_parameters()
        self.target_model.lstm.flatten_parameters()

    def reset_models(self):
        ...