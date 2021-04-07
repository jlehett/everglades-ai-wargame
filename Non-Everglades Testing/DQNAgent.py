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

# Temperature settings for Boltzmann exploration
TEMP_START = 1e+2
TEMP_END = 1.0
TEMP_DECAY = 0.00005

steps_done = 0

# Use custom reward shaping
custom_reward = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0)
has_gpu = torch.cuda.is_available()
#print(torch.cuda.version)

class DQNAgent():
    def __init__(self,action_space,observation_space, player_num, lr, replay_size, batch_size, 
                gamma, weight_decay, exploration, eps_start, eps_end,eps_decay,target_update):

        # Setup constants
        self.lr = lr
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.exploration = exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

        #Base Setup for the DQN Agent
        self.eps_threshold = 0
        self.Temp = 0
        self.action_space = action_space.n
        self.num_groups = 12

        self.n_actions = action_space.n
        self.n_observations = observation_space.shape
        self.seed = 1

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()

        # Added weight decay and LR for testing
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = self.lr, weight_decay=self.weight_decay)
        self.memory = ReplayMemory(self.replay_size)

        self.steps_done = 0

        self.num_nodes = 11
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)
        self.loss = 0
        self.q_values = np.zeros((2))
    
    def get_action(self, obs):
        obs = torch.from_numpy(obs).to(device)
        # Added new style of exploration for testing
        # Do not use
        if self.exploration == "Boltzmann":
            action = self.boltzmann(action, obs)
        elif self.exploration == "EPS":
            action = self.epsilon_greedy(obs)
        
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

    def epsilon_greedy(self, obs):
        global steps_done
        sample = random.random()
        ### Updated the eps equation to be more readable (based on the pytorch implementation on 
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(steps_done * -self.eps_decay)
        ###
        steps_done += 1

        if sample > self.eps_threshold:
            policy_out = self.policy_net(obs)
            self.q_values = policy_out.cpu().detach().numpy()
            action = policy_out.max(0)[1].item()
        else:
            action= np.random.randint(2)

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
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # No longer need mask with batched actions
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                    batch.next_state)), device=device, dtype=torch.bool)
        #

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.stack([s for s in batch.state])
        action_batch = torch.cat([s.unsqueeze(0) for s in batch.action])
        reward_batch = torch.cat([s.unsqueeze(0) for s in batch.reward])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)

        # Sets action_batch to be column-wise instead of row-wise for torch.batch()
        # Using Long() for indexing requirements per torch.batch()
        # No longer require unsqueezing with batched actions
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
        next_state_values = torch.zeros(self.batch_size, device=device)
        # Used topk() instead of max to grab the top 7 actions instead of the top 1 action
        next_state_values = self.target_net(non_final_next_states.view(self.batch_size,4)).max(1)[0].detach()
        
        # Compute the expected Q values
        # Floated the rewards to prevent errors
        # Added repeat to rewards so the tensor will line up with next_state_values for addition
        #reward_batch = reward_batch.unsqueeze(1).detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.type(torch.DoubleTensor), expected_state_action_values.type(torch.DoubleTensor).unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #Noisy net reset params
        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()

        # Sets the loss to be grabbed by training file
        self.loss = loss
    
    def update_target(self,episodes):
        # Updates the target model to reflect the current policy model
         if episodes % self.target_update == 0:
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
    def __init__(self,observation_size,action_size, seed, fc1_unit = 128,
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
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_size[0],fc1_unit)
        
        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        self.fc2 = nn.Linear(fc1_unit,action_size)

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################

        #self.fc2_adv = nn.Linear(fc1_unit,fc2_unit)
        #self.fc2_val = nn.Linear(fc1_unit,fc2_unit)

        #self.fc3_adv = nn.Linear(fc2_unit,action_size)
        #self.fc3_val = nn.Linear(fc2_unit,1)

        ##############################################################################################
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #print("x: ", x)
        #print("x-type", type(x))
        if(type(x).__module__ == np.__name__):
            x = torch.from_numpy(x).float()
        
        x = x.float()
        
        x = F.relu(self.fc1(x))
        
        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        x = self.fc2(x)

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################
        
        #adv = F.relu(self.fc2_adv(x))
        #val = F.relu(self.fc2_val(x))

        #adv = self.fc3_adv(adv)
        #val = self.fc3_val(val)

        #advAverage = adv.mean()
        #x = val + adv - advAverage

        ##############################################################################################

        return x

import math
import torch
from torch import nn
from torch.nn import init, Parameter
from torch.nn import functional as F
from torch.autograd import Variable

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
