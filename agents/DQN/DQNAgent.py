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

# PER
from pathlib import Path
from agents.DQN.PER import SumTree
from agents.DQN.PER import Memory
# from collections import deque

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05

### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_DECAY = 0.00005
###

TARGET_UPDATE = 10
steps_done = 0

# Use custom reward shaping
custom_reward = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        # PER
        self.memory = Memory(10000)
        # self.memory = deque(maxlen=10000)


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
        #if len(self.memory) < BATCH_SIZE:
            #return
    
        tree_idx, transitions = self.memory.sample(BATCH_SIZE)
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)
        #state_action_values = torch.gather(state_action_values,1.0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        hold = state_action_values.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # PER
        indices = np.arange(BATCH_SIZE, dtype=np.int32)
        # absolute_errors = np.abs(state_action_values[indices, np.array(action_tensor)]-expected_state_action_values[indices, np.array(action_tensor)])
        absolute_errors = np.abs(hold-expected_state_action_values)
        self.memory.batch_update(tree_idx, absolute_errors)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        self.memory.batch_update(tree_idx, absolute_errors)
    
    def update_target(self,episodes):
        # Updates the target model to reflect the current policy model
         if episodes % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, next_state, reward):
        experience = state, action, next_state, reward
        self.memory.store(experience)
        

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
    # PER hyperparameters
    PER_e = 0.01  # use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    # Simple Replay Memory (PER update)
    def __init__(self, capacity):
        self.tree = SumTree(capacity) # PER sumtree
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # Store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max priority for new priority

    # Old sample()
    def trans_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index
            print()
            minibatch.append([data[0],data[1],data[2],data[3],data[4]])

        return b_idx, minibatch

    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)



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
