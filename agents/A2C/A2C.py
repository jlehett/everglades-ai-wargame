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
device = 'cpu'

# Hyperparameters
learning_rate = 1e-2
gamma = 0.999
epsilon = 0.99

SavedAction = namedtuple('SavedAction', ['log_prob','value'])

class A2C():
    def __init__(self,action_space,observation_space, player_num, K_epochs):
        self.state_space = observation_space.shape[0]
        self.action_space = action_space
        self.K_epochs = K_epochs

        self.memory = Memory()
        
        self.shape = (self.action_space, 2)

        self.model = ActorCritic(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.model.parameters())

    def get_action(self, obs):
        action = np.zeros(self.shape)
        chosen_indices = self.model.act(obs, self.memory)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes

        #log_prob = self.model.evaluate(obs, action)
        return action

    def optimize_model(self):
        R = 0
        save_actions = self.model.save_actions
        policy_loss = []
        value_loss = []
        rewards = []
        values = []
        log_probs = []
        entropy = 0

        for r in self.model.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0,R)
        
        #rewards = torch.tensor(rewards)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)

        #for (log_prob, value), r in zip(save_actions, rewards):
            
        #    reward = r - value.item()
        #    policy_loss.append(-log_prob * reward)
        #    value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        ######### What we should be doing, but isn't currently working #########
        ########################################################################
        rewards   = torch.cat(rewards).detach()
        log_probs = torch.cat(log_probs)
        values    = torch.cat(values)

        advantage = rewards - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########################################################################

        print('policy loss', policy_loss)
        print('value loss', value_loss)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()

        del self.model.rewards[:]
        del self.model.save_actions[:]

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
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # First layer
        self.fc1 = nn.Linear(state_dim, 528)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 528),
            nn.Linear(528, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic, return a scalar value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 528),
            nn.Linear(528, action_dim)
        )

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        action_score = self.actor(x)
        state_value = self.critic(x)
        dist  = Categorical(probs)
        return action_score, state_value

    def act(self, state, memory):
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)
        
        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False)

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

        return action_logprobs, torch.squeeze(state_value), dist_entropy