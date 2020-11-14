# imports
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn # components of neural networks (nn), extend nn.module class
import torch.nn.functional as F
import torchvision.transforms as T
import json
# set up display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: 
    from IPython import display

class DM_DQN():
    def __init__(self, action_space, observation_space, player_num, map_name):
        self.current_step = 0
        self.num_actions = action_space
        self.batch_size = 256
        self.gamma = 0.999
        self.eps_start = 1
        self.eps_end = 0.3
        self.eps_decay = 0.99
        self.target_update = 10
        self.memory_size = 100000
        self.lr = 0.001
        self.num_episodes = 200
        self.num_groups = 12
        self.epsilon = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)

        self.replay = ReplayMemory(100000)
        self.policy_net = QNetwork(action_space, observation_space, 
                map_name, self.lr, self.epsilon, self.eps_decay, self.gamma)

        with open('D:\\Senior Design\\everglades-ai-wargame\\config\\' + map_name) as fid:
            self.map_dat = json.load(fid)


        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])

        self.num_nodes = len(self.map_dat['nodes'])

        self.shape = (self.num_actions, 2)

        self.unit_config = {
            0: [('controller',1), ('striker', 5)],# 6
            1: [('controller',3), ('striker', 3), ('tank', 3)],# 15
            2: [('tank',5)],# 20
            3: [('controller', 2), ('tank', 4)],# 26
            4: [('striker', 10)],# 36
            5: [('controller', 4), ('striker', 2)],# 42
            6: [('striker', 4)],# 46
            7: [('controller', 1), ('striker', 2), ('tank', 3)],# 52
            8: [('controller', 3)],# 55
            9: [('controller', 2), ('striker', 4)],# 61
            10: [('striker', 9)],# 70
            11: [('controller', 20), ('striker', 8), ('tank', 2)]# 100
        }

    def get_action(self, state):
        rate = self.epsilon.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = np.zeros(self.shape)
            action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            action[:, 1] = np.random.choice(self.nodes_array, self.num_actions, replace=False)
        else:
            with torch.no_grad():
                #return self.policy_net(state).argmax(dim = 1).item() # exploit
                action_hold = self.policy_net(state)
                action_hold = torch.reshape(action_hold, (12,11))
                action_units = np.zeros(7)
                action_nodes = np.zeros(7)
                action_qs = np.zeros(7)
                for i in range(12):
                    for j in range(11):
                        for k in range(7):
                            
                            if action_hold[i,j] > action_qs[k]:
                                action_qs[k] = action_hold[i,j]
                                action_units[k] = i
                                action_nodes[k] = j
                                break
                
                action[:, 0] = action_units
                action[:, 1] = action_nodes

        return action
        
    def extract_tensors(experiences):
        # Convert batch of Experiences to Experience of batches
        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)

        return (t1,t2,t3,t4)

class QNetwork(nn.Module):
    def __init__(self, action_space, observation_space, map_name, lr, epsilon, epsilon_decay,
    gamma):
        super().__init__()
        # linear layers
        self.fc1 = nn.Linear(in_features = observation_space, out_features =128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 128)
        self.out = nn.Linear(in_features = 128, out_features = 136)

    def forward(self, t):
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size