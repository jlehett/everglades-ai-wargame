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

class A2C_Modified():
    def __init__(self,action_space,observation_space, n_latent_var, K_epochs):
        self.state_space = observation_space.shape[0]
        self.action_space = action_space
        self.n_latent_var = n_latent_var
        self.K_epochs = K_epochs
        print(action_space)
        self.memory = Memory()
        self.actor_loss = 0
        self.critic_loss = 0
        self.shape = (7, 2)

        self.model = ActorCritic(self.state_space, self.action_space, self.n_latent_var)
        #self.actorModel = ActorNetwork(self.state_space, self.action_space, self.n_latent_var)
        #self.criticModel = CriticNetwork(self.state_space, self.action_space, self.n_latent_var)
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.model.parameters(),  lr=0.01)

    def get_action(self, obs):
        action = np.zeros(self.shape)
        # action = chosen_indices.item()
        chosen_indices = self.model.act(obs, self.memory)
  
        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes

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
            discounted_reward = reward + (gamma * discounted_reward)
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

            self.actor_optimizer.zero_grad()
            self.actor_loss = -(logprobs * advantage.detach()).mean()
            self.actor_loss.backward()
            torch.nn.utils.clip_grad_norm(actor.parameters(),0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            self.critic_loss = advantage.pow(2).mean()
            self.critic_loss.backward()
            torch.nn.utils.clip_grad_norm(critic.parameters(),0.5)
            self.critic_optimizer.step()

            #loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            # take gradient step
            #self.optimizer.zero_grad()
            #loss.mean().backward()
            #self.optimizer.step()
            #self.loss = loss

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

###########################
#   Actor Network Class   #
###########################
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorNetwork, self).__init__()

        # actor
        nn.Linear(state_dim, n_latent_var),
        #nn.Tanh(),
        nn.Linear(n_latent_var,n_latent_var),
        #nn.Tanh(),
        nn.Linear(n_latent_var, action_dim),
        #nn.Softmax(dim=-1)

    def forward(self, x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

###########################
#   Critic Network Class   #
###########################
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(CriticNetwork, self).__init__()

        nn.Linear(state_dim, n_latent_var),
        #nn.Tanh(),
        nn.Linear(n_latent_var,n_latent_var),
        #nn.Tanh(),
        nn.Linear(n_latent_var, 1)

    def forward(self, x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

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
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)
        
        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False) #do once

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