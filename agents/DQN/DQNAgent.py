import random
import numpy as np
from collections import namedtuple
from itertools import count
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F

# Import Rainbow Modules
from agents.DQN.NoisyLinear import NoisyLinear
from agents.DQN.PrioritizedMemory import PrioritizedMemory
from agents.DQN.SimpleMemory import ReplayMemory
from agents.DQN.QNetwork import QNetwork

steps_done = 0

# Use custom reward shaping
custom_reward = False

device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQNAgent():
    """
    Implementation of Deep Q Network for everglades
    """
    def __init__(self,
                action_space,
                observation_space, 
                player_num, 
                lr, 
                replay_size, 
                batch_size, 
                gamma, 
                eps_start, 
                eps_end,
                eps_decay,
                target_update,
                DEVICE = "CPU",
                train = True,
                save_after_episode = 100,
                network_save_name = None):
        """
        Initializes the DQN for everglades

        @param action_space The size of the actions in everglades
        @param observation_space The size of the observation space in everglades
        @param player_num The player number given by everglades
        @param lr The set learning rate
        @param replay_size The set size of the replay memory
        @param batch_size The size of the batches to be pulled from the replay memory
        @param gamma The discount factor in the loss function
        @param eps_start The starting value of epsilon
        @param eps_end The ending value of epsilon
        @param eps_decay The decay rate of epsilon
        @param target_update The time until an update of the target network occurs
        @param DEVICE The device to use during training/evaluation
        @param train Bool value to determine whether or not to train
        @param save_after_episode The episode number to save at. Ex curr_episode % save_after_episode == 0
        @param network_save_name The save file for the agent
        """

        # Enables GPU Training
        global device
        if DEVICE == 'GPU':
            device = torch.device('cuda')

        # Setup general parameters
        self.lr = lr
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.eps_threshold = 0

        # Set agent saving parameters
        self.network_save_name = network_save_name
        self.save_after_episode = save_after_episode

        #Base Setup for the DQN Agent
        self.action_space = action_space
        self.num_groups = 12
        self.n_actions = action_space
        self.n_observations = observation_space.shape
        self.seed = 1

        ## SETUP THE NETWORK ##
        self.policy_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net = QNetwork(self.n_observations,self.n_actions,self.seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.train = train

        # Create Optimizer and Memory
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr = self.lr)
        self.memory = ReplayMemory(self.replay_size)
        self.prioritized_memory = PrioritizedMemory(self.replay_size)

        # Miscellaneous class variables
        self.steps_done = 0
        self.num_nodes = 11
        self.num_actions = action_space
        self.shape = (self.num_actions, 2)
        self.loss = 0
        self.q_values = np.zeros((132))

        # Prioritized experience beta values
        self.beta_start = 0.4
        self.beta_frames = 1000 
        self.beta_by_frame = lambda frame_idx: min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def get_action(self, obs):
        """
        Handler method that gets an action from the QNetwork

        @param obs The observation by the agent in everglades
        """
        action = np.zeros(self.shape)
        obs = torch.from_numpy(obs).to(device)
        action = self.epsilon_greedy(action, obs)
        
        return action

    def epsilon_greedy(self, action, obs):
        """
        Gets an action according to the epsilon greedy approach

        @param action The action array to store the chosen action in
        @param obs The observatoin seen by the agent
        """

        global steps_done
        sample = random.random()

        ### Updated the eps equation to be more readable (based on the pytorch implementation on 
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(steps_done * -self.eps_decay)
        ###
        steps_done += 1

        if sample > self.eps_threshold:
            # Get greedy action
            action_qs = self.policy_net(obs)
            action = self.filter_actions(action, action_qs)
            self.q_values = action_qs
        else:
            # Get random action
            action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
            action[:, 1] = np.random.choice(self.num_nodes, self.num_actions, replace=False)

        return action

    def filter_actions(self, action, action_qs):
        """
        Filter the actions to get the actions with the highest possible q-values

        @param action The action array to store the chosen actions in
        @param action_qs The q-values for each possible action
        """
        with torch.no_grad():
            # Get the action_output from network and reshape to 2D tensor
            action_qs = torch.reshape(action_qs, (self.num_groups, self.num_nodes))

            # Initialize unit, node and q-value arrays
            best_action_units = np.zeros(self.n_actions)
            best_action_nodes = np.zeros(self.n_actions)
            best_action_qs = np.zeros(self.n_actions)

            # Unravel the output tensor into two size 7 arrays
            # Flipped node and group for loops to better match the reshape
            for node_index in range(self.num_nodes):
                for group_index in range(self.num_groups):
                    for action_index in range(self.n_actions):
                        # Get largest q-value actions
                        # Discard if lower than another action
                        if action_qs[group_index, node_index] > best_action_qs[action_index]:
                            # Prevent unit numbers from appearing in best_action_units multiple times
                            if group_index in best_action_units and best_action_units[action_index] != group_index:
                                continue
                            else:
                                best_action_qs[action_index] = action_qs[group_index, node_index]
                                best_action_units[action_index] = group_index
                                best_action_nodes[action_index] = node_index
                                break
            
            # Create the final action array to return in a readable format
            action[:, 0] = best_action_units
            action[:, 1] = best_action_nodes
        return action

    def remember_game_state(self, observation = None, action = None, reward = None, next_observation = None):
        """
        Stores game state information in the agent's memory

        @param observation The current observation seen by the agent
        @param action The action the agent took
        @param reward The reward the agent received
        @param next_observation The enxt observation the agent has seen
        """

        batch_actions = np.zeros(7)

        # Unwravel actions
        for i in range(7):
            batch_actions[i] = (action[i][0] * 11 + action[i][1])

        self.memory.push(torch.from_numpy(observation), 
                        torch.from_numpy(batch_actions), 
                        torch.from_numpy(next_observation), 
                        torch.from_numpy(np.asarray(reward)))

    def optimize_model(self):
        """
        Optimizes the DQN by taking a gradient step
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # No longer need mask with batched actions
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                    batch.next_state)), device=device, dtype=torch.bool)
        #

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.stack([s for s in batch.state])
        action_batch = torch.cat([s.unsqueeze(0) for s in batch.action])
        reward_batch = torch.cat([s.unsqueeze(0) for s in batch.reward])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)

        # Sets action_batch to be column-wise instead of row-wise for torch.batch()
        # Using Long() for indexing requirements per torch.batch()
        # No longer require unsqueezing with batched actions
        action_batch_unsqueezed = action_batch.long()#.unsqueeze(-1)

        # Pull out state action values that line up with previous actions
        # Check https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        # for reference on how it works
        state_action_values = torch.gather(state_action_values,1,action_batch_unsqueezed)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        # Used topk() instead of max to grab the top 7 actions instead of the top 1 action
        next_state_values = self.target_net(non_final_next_states.view(self.batch_size,105)).topk(7,1)[0]#.max(1)[0].detach()
        
        # Compute the expected Q values
        # Floated the rewards to prevent errors
        # Added repeat to rewards so the tensor will line up with next_state_values for addition
        reward_batch = reward_batch.unsqueeze(1).detach().repeat(1,7)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #Noisy net reset params
        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()

        # Sets the loss to be grabbed by training file
        self.loss = loss


    
    def prioritized_optimize_model(self, i_episode):
        """
        Optimizes the loss using prioritized experience replay

        @param i_episode The current episode number
        """
        # Calculate the beta
        beta = self.beta_by_frame(i_episode)
        # Sample from the memory
        state, action, reward, next_state, done, indices, weights = self.prioritized_memory.sample(self.batch_size, beta)

        # Setup data for calculations
        state      = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action     = torch.cat([s.unsqueeze(0) for s in action]).long()
        reward     = torch.FloatTensor(reward).unsqueeze(1).detach().repeat(1,7)
        done       = torch.FloatTensor(done).unsqueeze(1).detach().repeat(1,7)
        weights    = torch.FloatTensor(weights).unsqueeze(1).detach().repeat(1,7)

        # Get action q_vals
        q_values      = self.policy_net(state)
        # Get targeted vals
        next_q_values = self.target_net(next_state)

        # Correct remaining parts of loss function
        q_value          = torch.gather(q_values,1,action)
        next_q_value     = next_q_values.topk(7,1)[0]#.max(1)[0].detach()
        expected_q_value = reward + (self.gamma * next_q_value * (1 - done))
        
        loss  = (q_value - expected_q_value).pow(2) * weights
        prios = loss.mean(1) + 1e-5
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.prioritized_memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        #Noisy net reset params
        #self.policy_net.reset_noise()
        #self.target_net.reset_noise()

        # Sets the loss to be grabbed by training file
        self.loss = loss
    
    def update_target(self,episodes):
        """
        Updates the target network by copying in the weights of the policy network

        @param episodes The current episode the agent is on
        """
        # Updates the target model to reflect the current policy model
        if episodes % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_of_episode(self, episode):
        """
        Handles end of episode logic for the agent

        @param episode The current episode the agent is on
        """
        # Handle end of episode logic
        if self.train:
            self.end_of_episode_train(episode)
        

    def end_of_episode_train(self, episode):
        """
        Handles end of episode logic while agent is training

        @param episode The current episode the agent is on
        """
        # Handle end of episode while training
        if episode % self.save_after_episode == 0:
            self.save_network(episode)

    def save_network(self, episodes):
        """
        Saves the network's state dict, epsilon value, and episode count to the specified file.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        if self.network_save_name:
            save_file = open(self.network_save_name + '.pickle', 'wb')
            pickle.dump({
                'type': 'PPO',
                'policy_old_state_dict': self.policy_old.state_dict(),
                'policy_state_dict': self.policy.state_dict(),
                'eps_start': self.eps_start,
                'eps_end': self.eps_end,
                'eps_decay': self.eps_decay,
                'target_update': self.target_update,
                'replay_size': self.replay_size,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'gamma': self.gamma
            }, save_file)
            save_file.close()
            print('Saved Network')
        else:
            print('Save Failed - Save File Not Specified')