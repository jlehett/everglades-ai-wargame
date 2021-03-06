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

SAVE_NETWORK_AFTER = 5 # Save the network every n episodes

global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

# Hyperparameters

SavedAction = namedtuple('SavedAction', ['log_prob','value'])

class A2CAgent():
    def __init__(
        self,
        action_space,
        observation_space,
        n_latent_var,
        K_epochs,
        gamma,
        network_save_name = '/agents/A2CAgent/saved_models/A2C_test_2',
        network_load_name = None
    ):

        self.network_save_name = network_save_name
        self.network_load_name = network_load_name
        
        self.state_space = observation_space.shape[0]
        self.action_space = action_space
        self.n_latent_var = n_latent_var
        self.K_epochs = K_epochs
        self.gamma = gamma
        self.memory = Memory()
        self.loss = 0
        self.shape = (7, 2)

        self.model = ActorCritic(self.state_space, self.action_space, self.n_latent_var)
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters())

        if self.network_load_name and os.path.exists(self.network_load_name + '.pickle'):
            save_file = open(self.network_load_name + '.pickle', 'rb')
            save_file_data = pickle.load(save_file)
            save_file.close()

    def get_action(self, obs):
        action = np.zeros(self.shape)
        chosen_indices = self.model.act(obs, self.memory)
  
        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units.cpu()
        action[:,1] = chosen_nodes.cpu()

        #log_prob = self.model.evaluate(obs, action)
        #print(action)
        return action

    #def calc_return()

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        entropy = 0
      
        # Calculate reward discounts 
        for reward in reversed(self.memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
       
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        states = torch.stack(self.memory.states).to(device).detach()
        actions = torch.stack(self.memory.actions).to(device).detach()
        logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.model.evaluate(states, actions)
            entropy += dist_entropy.mean()
            
        advantage = rewards - state_values

        actor_loss = -(logprobs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        self.loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.memory.clear_memory()

    def save_network(self, episodes):
        """
        Saves the network's state dict, epsilon value, and episode count to the specified file.
        @param episodes The number of episodes that have elapsed since the current training session began
        """
        if self.network_save_name:
            save_file = open(os.getcwd() + self.network_save_name + '.pickle', 'wb')
            pickle.dump({
                'model_state_dict': self.model.state_dict(),
                'n_latent_var': self.n_latent_var,
                'k_epochs': self.K_epochs,
                'gamma': self.gamma,
            }, save_file)
            save_file.close()
            print('Saved Network')
        else:
            print('Save Failed - Save File Not Specified')

####################
#   Memory Class   #
####################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

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

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)
        
        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False).to(device)

        for i in range(7):
            memory.logprobs.append(dist.log_prob(action_indices[i]))
            memory.states.append(state)
            memory.actions.append(action_indices[i])

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
