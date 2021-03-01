## Dave Mahoney A2C ##
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# Hyperparameters
learning_rate = 1e-2
gamma = 0.999
epsilon = 0.99
hidden_layer_size = 528

SavedAction = namedtuple('SavedAction', ['log_prob','value'])

class A2C():
    def __init__(self,action_space,observation_space, player_num):
        self.state_space = observation_space.shape[0]
        self.action_space = action_space
        
        self.shape = (self.action_space, 2)

        self.model = ActorCritic(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.model.parameters())

    def get_action(self, obs):
        action = np.zeros(self.shape)
        chosen_indices = self.model.act(obs)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes

        log_prob = evaluate(action)
        
        return action

    def optimize_model(self):
        R = 0
        save_actions = model.save_actions
        policy_loss = []
        value_loss = []
        rewards = []
        
        for r in model.rewards[::-1]:
            R = r + gamma * R
            rewrds.insert(0,R)
        
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)

        for (log_prob, value), r in zip(save_actions, rewards):
            reward = r - value.item()
            policy_loss.append(-log_prob * reward)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()

        del model.rewards[:]
        del model.save_actions[:]

#########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # First layer
        #self.fc1 = nn.Linear(state_dim, hidden_layer_size)

        # actor
        # self.actor = nn.Linear(hidden_layer_size, action_dim)

        # critic, return a scalar value
        #self.critic = nn.Linear(hidden_layer_size, 1) 

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_dim),
            nn.Softmax(dim=-1), # dim should be either -1 or 0, not sure which but PPO has -1
        )

        self.policy_action_value = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.actor(x)
        state_value = self.critic(x)

        return F.softmax(action_score, dim=-1), state_value

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)

        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False)

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

        return action_logprobs