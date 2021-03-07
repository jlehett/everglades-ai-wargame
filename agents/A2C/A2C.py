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
    def __init__(self,action_space,observation_space, K_epochs):
        self.state_space = observation_space.shape[0]
        self.action_space = action_space
        self.K_epochs = K_epochs

        self.memory = Memory()
        
        self.shape = (self.action_space, 2)

        self.model = ActorCritic(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.model.parameters())

    def get_action(self, obs):
        action = np.zeros((7, 2))

        chosen_indices = self.model.act(obs, self.memory)

        # Unravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes

        #print(action)

        #log_prob = self.model.evaluate(obs, action)
        return action

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
      
        # Calculate reward discounts 
        for reward in reversed(self.memory.rewards):
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
       

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        #print(rewards)

        #print(rewards.size())

        # convert list to tensor
        states = torch.stack(self.memory.states).to(device).detach()
        actions = torch.stack(self.memory.actions).to(device).detach()
        logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.model.evaluate(states, actions)

        #print(logprobs.size())
        #print(state_values)
        #print(dist_entropy.size())

        advantage = rewards - state_values.detach()

        #print("Advantage:")
        #print(advantage)
        #print(advantage.size())
        #print(-(logprobs * advantage.detach()).mean())

        actor_loss = -(logprobs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        #print("Actor loss:")
        #print(actor_loss)
        #print("Critic loss:")
        #print(critic_loss)
        #print("")

        self.loss = actor_loss + 0.5 * critic_loss - 0.001 * dist_entropy.mean()
        #print(loss.size())
        #for r in self.model.rewards[::-1]:
            #R = r + gamma * R
            #rewards.insert(0,R)
        
        #rewards = torch.tensor(rewards)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)

        #for (log_prob, value), r in zip(save_actions, rewards):
            
            #reward = r - value.item()
            #policy_loss.append(-log_prob * reward)
            #value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.memory.clear_memory()

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
            nn.Tanh(),
            nn.Linear(528, 528),
            nn.Tanh(),
            nn.Linear(528, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic, return a scalar value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 528),
            nn.Tanh(),
            nn.Linear(528, 528),
            nn.Tanh(),
            nn.Linear(528, 1)
        )

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        action_score = self.actor(x)
        state_value = self.critic(x)
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
        #print(state_value.size())

        return action_logprobs, torch.squeeze(state_value), dist_entropy