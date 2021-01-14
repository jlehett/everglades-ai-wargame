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

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05

### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_DECAY = 0.00005
###

TARGET_UPDATE = 10
steps_done = 0

# Distributional RL variables
N_ATOM = 51
V_MIN = -5.
V_MAX = 10.
V_RANGE = np.linspace(V_MIN, V_MAX, N_ATOM)
V_STEP = ((V_MAX-V_MIN)/(N_ATOM-1))

# Use custom reward shaping
custom_reward = False

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class DQNAgent():
    def __init__(self,action_space,observation_space, player_num,map_name):
        #Base Setup for the DQN Agent
        self.eps_threshold = 0
        self.action_space = action_space
        self.num_groups = 12

        self.n_actions = action_space
        self.n_observations = observation_space.shape
        self.seed = 1

        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)

        # discrete values for DRL
        self.value_range = torch.FloatTensor(V_RANGE) # (N_ATOM)

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])

        self.num_nodes = len(self.map_dat['nodes'])
        self.num_actions = action_space

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
    
    def get_action(self, obs):
        global steps_done
        sample = random.random()

        ### Updated the eps equation to be more readable (based on the pytorch implementation on 
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(steps_done * -EPS_DECAY)
        ###

        #print("Current Epsilon: {}\tSteps Done: {}\n", eps_threshold,steps_done)
        steps_done += 1
        action = np.zeros(self.shape)
        if sample > self.eps_threshold:
            with torch.no_grad():

                # Get the action_output from network and reshape to 2D tensor
                action_hold = self.policy_net(obs)
                action_hold = torch.reshape(action_hold, (12,11))

                # Initialize unit, node and q-value arrays
                action_units = np.zeros(7)
                action_nodes = np.zeros(7)
                action_qs = np.zeros(7)

                # Unravel the output tensor into two size 7 arrays
                for i in range(12):
                    for j in range(11):
                        for k in range(7):
                            # Get largest q-value actions
                            # Discard if lower than another action
                            if action_hold[i,j] > action_qs[k]:
                                action_qs[k] = action_hold[i,j]
                                action_units[k] = i
                                action_nodes[k] = j
                                break
                
                action[:, 0] = action_units
                action[:, 1] = action_nodes
        else:
            # Choose random action
            # Based on implementation in random_actions agent
            action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            action[:, 1] = np.random.choice(self.nodes_array, self.num_actions, replace=False)
            
        #print(action)
        return action

    ### Duplicate checker
    # Only tells the get_action not to use a particular unit
    def check_duplicates(self,i,j,action_units,action_nodes):
        no_dupes = True
        # Only need to check for duplicate units
        # Duplicate nodes are allowed
        for u in action_units:
            if u == i:
                no_dupes = False
        
        return no_dupes
    ###


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        next_state_tensor = torch.from_numpy(np.asarray(batch.next_state))
        state_tensor = torch.from_numpy(np.asarray(batch.state))
        action_tensor = torch.from_numpy(np.asarray(batch.action))
        reward_tensor = torch.from_numpy(np.asarray(batch.reward))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = next_state_tensor
        state_batch = state_tensor
        action_batch = action_tensor
        reward_batch = reward_tensor
        done_batch = non_final_mask

        weight_batch = np.ones_like(reward_batch)

        # action value distribution prediction
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)
        mb_size = state_action_values.size(0)
        #state_action_values = torch.stack([state_action_values[i].index_select(0, action_batch[i]) for i in range(mb_size)]).squeeze(1) 
        #state_action_values = torch.gather(state_action_values,1.0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        #print(next_state_values)

        # distribution of next_state_values, and next_state_value_mean
        #next_state_values_mean = torch.sum(next_state_values * self.value_range.view(1, 1, -1), dim=2) # (m, N_ACTIONS)
        #best_actions = next_state_values_mean.argmax(dim=1) # (m)
        #print(best_actions)
        #next_state_values = torch.stack([next_state_values[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1) 

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # target distribution
        q_target = np.zeros((mb_size, N_ATOM)) # (m, N_ATOM)

        # categorical projection
        '''
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        '''
        # we vectorized the computation of support and position
        next_v_range = np.expand_dims(reward_batch, 1) + GAMMA * np.expand_dims((~done_batch),1) \
        * np.expand_dims(self.value_range.data.cpu().numpy(),0) #what is value_range.data?
        #print(next_v_range)
        next_v_pos = np.zeros_like(next_v_range)
        # clip for categorical distribution
        next_v_range = np.clip(next_v_range, V_MIN, V_MAX)
        # calc relative position of possible value
        next_v_pos = (next_v_range - V_MIN)/ V_STEP
        # get lower/upper bound of relative position
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        #print(next_v_pos)
        #print(next_v_range)
        # we didn't vectorize the computation of target assignment.
        for i in range(mb_size):
            for j in range(N_ATOM):
                # calc prob mass of relative position weighted with distance
                q_target[i, lb[i,j]] += (next_state_values * (ub - next_v_pos))[i,j]
                q_target[i, ub[i,j]] += (next_state_values * (next_v_pos - lb))[i,j]

        q_target = torch.FloatTensor(q_target)

        # Calculate Huber loss, dont reduce for importance weight
        #loss = q_target * ( - torch.log(state_action_values + 1e-8)) # (m , N_ATOM)
        #loss = torch.mean(loss)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        #weight_batch = torch.Tensor(weight_batch)
        #loss = torch.mean(weight_batch*loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
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

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,observation_size,action_size, seed, fc1_unit = 528,
                 fc2_unit = 256, fc3_unit = 256):
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
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)

        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        self.fc3 = nn.Linear(fc2_unit, fc3_unit)
        self.fc4 = nn.Linear(fc3_unit, self.action_size)

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################

        #self.fc3_adv = nn.Linear(fc2_unit,fc3_unit)
        #self.fc3_val = nn.Linear(fc2_unit,fc3_unit)

        #self.fc4_adv = nn.Linear(fc3_unit,self.action_size)
        #self.fc4_val = nn.Linear(fc3_unit,1)

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
        x = F.relu(self.fc2(x))

        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

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
