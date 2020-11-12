import gym
import math
import random
import numpy as np
import json
from collections import namedtuple, deque
from itertools import count

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQNAgent():
    def __init__(self, action_space, observation_space, player_num, map_name):
        # Base Setup for the DQN Agent
        self.action_space = action_space
        self.num_groups = 12

        self.n_actions = action_space
        self.n_observations = observation_space.shape
        self.seed = 1

        self.epsilon = 0.3
        self.epsilon_decay = 0.9

        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)

        # Set Up the Network
        self.policy_net = QNetwork(self.n_actions, self.n_observations, self.seed).to(device)
        self.target_net = QNetwork(self.n_actions, self.n_observations, self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])

        self.num_nodes = len(self.map_dat['nodes'])
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)

    def get_random_action(self):
        action = np.zeros(self.shape)
        action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
        action[:, 1] = np.random.choice(self.nodes_array, self.num_actions, replace=True)
        return action

    def get_greedy_action(self, obs):
        action = np.zeros(self.shape)
        with torch.no_grad():
            action_hold = self.policy_net(obs)
            action_hold = torch.reshape(action_hold, (12, 11)).numpy()
            action_units = np.zeros(7)
            action_nodes = np.zeros(7)
            action_qs = np.zeros(7)
            for i in range(12):
                for j in range(11):
                    for k in range(7):
                        if action_hold[i,j] > action_qs[k]:
                            action_qs[k] = action_hold[i,j]
                            action_units[k] = i
                            action_units[k] = j
                            break
            action[:, 0] = action_units
            action[:, 1] =  action_nodes
        return action

    def get_action(self, obs):
        sample = random.random()
        if sample > self.epsilon:
            return self.get_greedy_action(obs)
        else:
            return self.get_random_action()

    def train(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
    ):
        self.memory.push(previous_state, actions, next_state, reward)

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        next_state_tensor = torch.from_numpy(np.asarray(batch.next_state))
        state_tensor = torch.from_numpy(np.asarray(batch.state))
        action_tensor = torch.from_numpy(np.asarray(batch.action))
        reward_tensor = torch.from_numpy(np.asarray(batch.reward))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = next_state_tensor
        state_batch = state_tensor
        action_batch = action_tensor
        reward_batch = reward_tensor

        state_action_values = self.policy_net(state_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def end_of_episode(self, episodes):
        self.epsilon *= self.epsilon_decay
        print(self.epsilon)
        if episodes % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, action_size, observation_size, seed, fc1_unit=128,
                 fc2_unit=128, fc3_unit=132):
        action_size = 12*11
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_size[0], fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, fc3_unit)
        self.fc4 = nn.Linear(fc3_unit, action_size)

    def forward(self, x):
        if (type(x).__module__ == np.__name__):
            x = torch.from_numpy(x)
        
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)