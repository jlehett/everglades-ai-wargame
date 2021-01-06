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
from agents.Multi_Step_Learning.Multi_Step import NStepModule

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
N_STEP = 16
MEMORY_SIZE = 10000

### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_DECAY = 0.00005
###

TARGET_UPDATE = 4

# Use custom reward shaping
custom_reward = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    def __init__(self, action_space, observation_space, player_num, map_name):
        #Base Setup for the DQN Agent
        self.num_groups = 12
        self.n_actions = action_space
        self.n_observations = observation_space.shaped
        self.shape = (self.n_actions, 2)
        self.steps_done = 0

        # Create the NStepModule
        self.NStepModule = NStepModule(N_STEP, GAMMA, MEMORY_SIZE)

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations, self.n_actions).to(device)
        self.target_net = QNetwork(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.no_grad()

        # Set the optimizer to use in training the network
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        # Set up map data to use in selecting random actions
        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)
            self.nodes_array = []
            for i, in_node in enumerate(self.map_dat['nodes']):
                self.nodes_array.append(in_node['ID'])
            self.num_nodes = len(self.map_dat['nodes'])
    
    def get_action(self, obs):
        sample = random.random()

        ### Updated the eps equation to be more readable (based on the pytorch implementation on 
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(self.steps_done * -EPS_DECAY)
        ###

        #print("Current Epsilon: {}\tSteps Done: {}\n", eps_threshold, self.steps_done)
        self.steps_done += 1
        action = np.zeros(self.shape)
        if sample > self.eps_threshold:
            with torch.no_grad():

                # Get the action_output from network and reshape to 2D tensor
                action_qs = self.policy_net(obs)
                action_qs = torch.reshape(action_qs, (self.num_groups, self.num_nodes))

                # Initialize unit, node and q-value arrays
                best_action_units = np.zeros(self.n_actions)
                best_action_nodes = np.zeros(self.n_actions)
                best_action_qs = np.zeros(self.n_actions)

                # Unravel the output tensor into two size 7 arrays
                for group_index in range(self.num_groups):
                    for node_index in range(self.num_nodes):
                        for action_index in range(self.n_actions):
                            # Get largest q-value actions
                            # Discard if lower than another action
                            if action_qs[group_index, node_index] > action_qs[action_index]:
                                # Prevent unit numbers from appearing in best_action_units multiple times
                                if group_index in best_action_units and best_action_units[action_index] !== group_index:
                                    continue
                                else:
                                    best_action_qs[action_index] = action_qs[group_index, node_index]
                                    best_action_units[action_index] = group_index
                                    best_action_nodes[action_index] = node_index
                                    break
                
                # Create the final action array to return in a readable format
                action[:, 0] = action_units
                action[:, 1] = action_nodes
        else:
            # Choose random action
            # Based on implementation in random_actions agent
            action[:, 0] = np.random.choice(self.num_groups, self.n_actions, replace=False)
            action[:, 1] = np.random.choice(self.nodes_array, self.n_actions, replace=False)
            
        #print(action)
        return action

    def optimize_model(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
    ):
        # Add the new experience to the NStepModule for experience replay
        self.NStepModule.trackGameState(previous_state, actions, reward)

        # If the NStepModule's experience replay isn't large enough, we should bail out.
        # Otherwise, we can grab sample data from the replay memory.
        if not self.NStepModule.isMemoryLargeEnoughToTrain(BATCH_SIZE):
            return
        transitions = self.NStepModule.sampleReplayMemory(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        nth_next_state_tensor = torch.from_numpy(np.asarray(batch.next_state))
        state_tensor = torch.from_numpy(np.asarray(batch.state))
        action_tensor = torch.from_numpy(np.asarray(batch.action))
        reward_tensor = torch.from_numpy(np.asarray(batch.reward))
        reward_tensor = torch.from_numpy(np.asarray(batch.reward).reshape((BATCH_SIZE, 1)))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = batch.hitsDone
        #print(non_final_mask)
        non_final_next_states = nth_next_state_tensor[non_final_mask]
        state_batch = state_tensor
        action_batch = action_tensor
        reward_batch = reward_tensor

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)
        #state_action_values = torch.gather(state_action_values,1.0, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros((BATCH_SIZE, self.num_groups*self.num_nodes), device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).detach()
        # Construct the appropriate training data
        next_state_values = next_state_values.numpy()
        max_next_state_value_per_batch = np.amax(next_state_values, axis=1)
        for i in range(next_state_values.shape[0]):
            next_state_values[i, :] = max_next_state_value_per_batch[i]
        next_state_values = torch.from_numpy(next_state_values)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * (GAMMA ** N_STEP)) + reward_batch
        expected_state_action_values = expected_state_action_values.type(torch.FloatTensor)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def end_of_episode(self, episodes):
        # Add the played game to memory
        self.NStepModule.addGameToReplayMemory()
        # Update target network every UPDATE_TARGET_AFTER episodes
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

### DEFINE REPLAY MEMORY TRANSITION ###

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hitsDone'))

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,observation_size,action_size, fc1_unit = 528,
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
        self.seed = torch.manual_seed(1)
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
