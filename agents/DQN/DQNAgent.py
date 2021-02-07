import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

LR = 1e-2
BATCH_SIZE = 150 # Updated
GAMMA = 0.999
LEAKY_SLOPE = 0.01 # Updated
WEIGHT_DECAY = 0 # Leave at zero. Weight decay has so far caused massive collapse in network output
EXPLORATION = "EPS" # Defines the exploration type. EPS is Epsilon Greedy, Boltzmann is Boltzmann Distribution Sampling

# Temperature settings for Boltzmann exploration
TEMP_START = 1e+2
TEMP_END = 1.0
TEMP_DECAY = 0.00005

### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.00005
###

TARGET_UPDATE = 100 # Updated
steps_done = 0

# Use custom reward shaping
custom_reward = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(torch.cuda.version)

class DQNAgent():
    def __init__(self,action_space,observation_space, player_num):
        #Base Setup for the DQN Agent
        self.eps_threshold = 0
        self.Temp = 0
        self.action_space = action_space
        self.num_groups = 12

        self.n_actions = action_space
        self.n_observations = observation_space.shape
        self.seed = 1

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Added weight decay and LR for testing
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = LR, weight_decay=WEIGHT_DECAY)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.num_nodes = 11
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)
        self.loss = 0
    
    def get_action(self, obs):
        action = np.zeros(self.shape)

        # Added new style of exploration for testing
        # Do not use
        if EXPLORATION == "Boltzmann":
            action = self.boltzmann(action, obs)
        elif EXPLORATION == "EPS":
            action = self.epsilon_greedy(action, obs)
        
        return action


    def boltzmann(self, action, obs):
        global steps_done
        sample = random.random()

        ### Borrowed EPS decay for Boltzmann temperature decay
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.Temp = TEMP_END + (TEMP_START - TEMP_END) * np.exp(steps_done * -TEMP_DECAY)
        ###

        steps_done += 1
        action_qs = self.policy_net(obs)
        action_qs = F.softmax(action_qs/self.Temp, dim=-1)

        # Sample actions using Boltzmann distribution without replacement
        action_indices = torch.multinomial(action_qs,7,replacement=False)

        # Unwravel chosen indices for action
        chosen_units = action_indices // 12
        chosen_nodes = action_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes
        return action

    def epsilon_greedy(self, action, obs):
        global steps_done
        sample = random.random()
        ### Updated the eps equation to be more readable (based on the pytorch implementation on 
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(steps_done * -EPS_DECAY)
        ###

        steps_done += 1

        if sample > self.eps_threshold:
            action_qs = self.policy_net(obs)
            action = self.filter_actions(action, action_qs)
        else:
            action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            action[:, 1] = np.random.choice(self.num_nodes, self.num_actions, replace=False)
        return action

    def filter_actions(self, action, action_qs):
        with torch.no_grad():
            # Get the action_output from network and reshape to 2D tensor
            action_qs = torch.reshape(action_qs, (self.num_groups, self.num_nodes))

            # Initialize unit, node and q-value arrays
            best_action_units = np.zeros(self.n_actions)
            best_action_nodes = np.zeros(self.n_actions)
            best_action_qs = np.zeros(self.n_actions)

            # Unravel the output tensor into two size 7 arrays
            # Flipped node and group for loops to better match the reshape
            for node_index in range(self.num_nodes):
                for group_index in range(self.num_groups):
                    for action_index in range(self.n_actions):
                        # Get largest q-value actions
                        # Discard if lower than another action
                        if action_qs[group_index, node_index] > best_action_qs[action_index]:
                            # Prevent unit numbers from appearing in best_action_units multiple times
                            if group_index in best_action_units and best_action_units[action_index] != group_index:
                                continue
                            else:
                                best_action_qs[action_index] = action_qs[group_index, node_index]
                                best_action_units[action_index] = group_index
                                best_action_nodes[action_index] = node_index
                                break
            
            # Create the final action array to return in a readable format
            action[:, 0] = best_action_units
            action[:, 1] = best_action_nodes
        return action

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state
                                                if s is not None])
        state_batch = torch.from_numpy(np.asarray(batch.state))
        action_batch = torch.from_numpy(np.asarray(batch.action))
        reward_batch = torch.from_numpy(np.asarray(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)

        # Sets action_batch to be column-wise instead of row-wise for torch.batch()
        # Using Long() for indexing requirements per torch.batch()
        action_batch_unsqueezed = action_batch.long().unsqueeze(-1)

        # Pull out state action values that line up with previous actions
        # Check https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # for reference on how it works
        state_action_values = torch.gather(state_action_values,1,action_batch_unsqueezed)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states.view(BATCH_SIZE,105)).max(1)[0].detach()
        
        # Compute the expected Q values
        # Floated the rewards to prevent errors
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Sets the loss to be grabbed by training file
        self.loss = loss
    
    def update_target(self,episodes):
        # Updates the target model to reflect the current policy model
         if episodes % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    #######################################
    # TODO: Internalized reward function  #
    #######################################
    def set_reward(self, obs):
        if(not custom_reward):
            return 0
        return
    #################################


### DEFINE REPLAY MEMORY ###

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    # Simple Replay Memory
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,observation_size,action_size, seed, fc1_unit = 528,
                 fc2_unit = 128):
        """
        Initialize parameters and build model.
        Params
        =======
            observation_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        self.action_size = 12 * 11
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_size[0],fc1_unit)

        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        self.fc2 = nn.Linear(fc1_unit,self.action_size)

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################

        #self.fc3_adv = nn.Linear(fc1_unit,fc2_unit)
        #self.fc3_val = nn.Linear(fc1_unit,fc2_unit)

        #self.fc4_adv = nn.Linear(fc2_unit,self.action_size)
        #self.fc4_val = nn.Linear(fc2_unit,1)

        ##############################################################################################
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #print("x: ", x)
        #print("x-type", type(x))
        if(type(x).__module__ == np.__name__):
            x = torch.from_numpy(x)
        
        x = x.float()
        x = F.relu(self.fc1(x))

        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        x = F.relu(self.fc2(x))

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################
        
        #adv = F.relu(self.fc3_adv(x))
        #val = F.relu(self.fc3_val(x))

        #adv = self.fc4_adv(adv)
        #val = self.fc4_val(val)

        #advAverage = adv.mean()
        #x = val + adv - advAverage

        ##############################################################################################

        return x
