## Dave Mahoney, Joel Membribe A2C ##
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import json
import pickle
import os
import random

global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

# Hyperparameters

SavedAction = namedtuple('SavedAction', ['log_prob','value'])

INPUT_SIZE = 105
OUTPUT_SIZE = 132

class A2C():
    def __init__(
        self,
        save_file_data,
        player_num,
        map_name,
    ):
        # Store the save file data
        self.save_file_data = save_file_data

        # Load the saved net
        self.model = ActorCritic(INPUT_SIZE, OUTPUT_SIZE, save_file_data['n_latent_var'])
        self.model.load_state_dict(save_file_data['model_state_dict'])

        # Final action shape
        self.shape = (7, 2)

    def get_action(self, obs):
        action = np.zeros(self.shape)
        chosen_indices = self.model.act(obs)
  
        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units.cpu()
        action[:,1] = chosen_nodes.cpu()

        #log_prob = self.model.evaluate(obs, action)
        #print(action)
        return action

    #def calc_return()

#########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
            #nn.Linear(state_dim, 528),
            #nn.Linear(528, action_dim),
            #nn.Softmax(dim=-1)
        )

        # critic, return a scalar value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
            #nn.Linear(state_dim, 528),
            #nn.Linear(528, 1)
        )


    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)
        
        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False).to(device)

        return action_indices

    def evaluate(self, state, action):
        action_probs = self.actor(state)

        # Use same distribution as act
        dist = Categorical(action_probs)

        # Calculate the expected log_probs for the previous actions
        action_logprobs = dist.log_prob(action)

        # Calculate the entropy from the distribution
        dist_entropy = dist.entropy()

        # Get expected network output
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value).to(device), dist_entropy
