# imports
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn # components of neural networks (nn), extend nn.module class
import torch.nn.functional as F
import torchvision.transforms as T
# set up display
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: 
    from IPython import display

# nn.Module is the base class for  NNs, DQN extends module
class Agent(nn.Module):

    def __init__(self, env, map_name, h, lr, epsilon, epsilon_decay=0.99,
    discount):
        super().__init__()
        # linear layers
        self.fc1 = nn.Linear(in_features = observation, out_features = 24)
        self.fc2 = nn.Linear(in_features = 24, out_features = 32)
        self.out = nn.Linear(in_features = 32, out_features = 136)

    def forward(self, t):
        t = t.flatten(start_dim = 1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

Experience = namedtuple(
    'Experience',
    ('state','action','next_state','reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class DM_DQN():
    def __init__(self, nn.Module, env, map_name, h, lr, epsilon, epsilon_decay=0.99,
    discount):
        #self.current_step = 0
        #self.strategy = strategy
        #self.num_actions = num_actions
        #self.epsilon = 
        batch_size = 256
        gamma = 0.999
        eps_start = 1
        eps_end = 0.3
        eps_decay = 0.99
        target_update = 10
        memory_size = 100000
        lr = 0.001
        num_episodes = 200

    def get_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions) #explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim = 1).item() # exploit

        
    def extract_tensors(experiences):
        # Convert batch of Experiences to Experience of batches
        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)

        return (t1,t2,t3,t4)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    
    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def train(self,
        previous_state,
        next_state,
        actions,
        reward)
    :
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.3
    eps_decay = 0.99
    target_update = 10
    memory_size = 100000
    lr = 0.001
    num_episodes = 200

    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DM_DQN(previous_state)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()
    
        for timestep in count():
            self.action = agent.select_action(state, policy_net)
            self.reward = em.take_action(action)
            self.next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)
            
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()