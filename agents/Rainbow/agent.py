
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import random
import numpy as np
from collections import namedtuple
import pickle
import os
from typing import Deque, Dict, List, Tuple
from torch.nn.utils import clip_grad_norm_

from agents.Rainbow.agent_attributes.PER import PrioritizedReplayBuffer
from agents.Rainbow.agent_attributes.Network import Network
from agents.Rainbow.agent_attributes.ReplayMemory import ReplayBuffer

TRAIN = False # If set to true, will use standard training procedure; if set to false, epsilon is ignored and the agent no longer trains
EVALUATE_EPSILON = 0.00 # The epsilon value to use when evaluating the network (when TRAIN is set to False)
TRAIN_EPSILON_START = 0.95 # The epsilon value to use when starting to train the network (when TRAIN is set to True)
TRAIN_EPSILON_MIN = 0.05 # The minimum epsilon value to use during training (when TRAIN is set to True)

NETWORK_SAVE_NAME = 'agents/Minimized_Rainbow/PerSwarm' # The name to use in saving the trained agent
NETWORK_LOAD_NAME = 'agents/Minimized_Rainbow/PerSwarm' # The name to use in loading a saved agent
#NETWORK_LOAD_NAME = None # The name to use in loading a saved agent
SAVE_NETWORK_AFTER = 10 # Save the network every n episodes

NUM_GROUPS = 12 # The number of unit groups in the Everglades environment for the agent
NUM_ACTIONS = 7 # The number of actions an agent can take in a single turn
EVERGLADES_ACTION_SIZE = (NUM_ACTIONS, 2) # The action shape in an Everglades-readable format

INPUT_SIZE = 59 # This is a custom value defined when creating the minimized input
OUTPUT_SIZE = 11 # This is the same as the number of nodes
FC1_SIZE = 80 # Number of nodes in the first hidden layer

BATCH_SIZE = 256 # The number of inputs to train on at one time
TARGET_UPDATE = 100 # The number of episodes to wait until we update the target network
MEMORY_SIZE = 10000 # The number of experiences to store in memory replay
GAMMA = 0.99 # The amount to discount the future rewards by
ALPHA = 0.2
LEARNING_RATE = 1e-4 # The learning rate to be used by the optimizer
N_STEP = 3 # The number of steps to use in multi-step learning
EPS_DECAY = 0.999 # The rate at which epsilon decays at the end of each episode

# distributional
V_MIN = 0.0
V_MAX = 200.0
ATOM_SIZE = 51

# PER
BETA = 0.6
PRIOR_EPS = 1e-6

TARGET_UPDATE = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQNAgent():
    def __init__(
        self,
        player_num,
        map_name,
        observation_space
    ):
        """
        Initialize a DQNAgent that will be used to play the Everglades game.
        Initializes class variables, creates policy and target networks, and
        loads the map data.

        @param player_num The player number of the agent in the Everglades environment
        @param map_name The name of the map file to load in
        """
        # Store variables
        self.device = device

        self.update_count = 0

        if TRAIN:
            self.epsilon = TRAIN_EPSILON_START
        else:
            self.epsilon = EVALUATE_EPSILON
        self.previous_episodes = 0
        self.training = TRAIN

        # Create the NStepModule
        #self.NStepModule = NStepModule(N_STEP, GAMMA, MEMORY_SIZE)
        if N_STEP > 1:
            self.NStepMemory = ReplayBuffer(INPUT_SIZE, MEMORY_SIZE, BATCH_SIZE, N_STEP, GAMMA)

        # PER memory
        self.ReplayMemory = PrioritizedReplayBuffer(INPUT_SIZE, MEMORY_SIZE, BATCH_SIZE, ALPHA)
        self.beta = BETA
        self.prior_eps = PRIOR_EPS

        self.transition = list()

        # Set up the network
        #self.policy_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE)
        #self.target_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE)
        self.support = torch.linspace(V_MIN, V_MAX, FC1_SIZE).to(device)
        self.policy_net = Network(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE, self.support)
        self.target_net = Network(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE, self.support)

        # # If a save file is specified and the file exists, load the save file
        # if NETWORK_LOAD_NAME and os.path.exists(NETWORK_LOAD_NAME + '.pickle'):
        #     save_file = open(NETWORK_LOAD_NAME + '.pickle', 'rb')
        #     save_file_data = pickle.load(save_file)
        #     self.policy_net.load_state_dict(save_file_data.get('state_dict'))
        #     if TRAIN:
        #         self.epsilon = save_file_data.get('epsilon')
        #         self.previous_episodes = save_file_data.get('episodes')
        #     save_file.close()
        #     print('Loaded Saved Network:', NETWORK_LOAD_NAME)

        # Load the policy network's values into the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Prevent the target net from learning during training; only the policy
        # net should be learning
        self.target_net.eval()
        self.num_episodes = 0

        # Set the optimizer to use in training the network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Set up the map data to use
        with open(map_name) as fid:
            self.map_dat = json.load(fid)
            self.nodes_map = {}
            for i, in_node in enumerate(self.map_dat['nodes']):
                self.nodes_map[in_node['ID']] = in_node
            self.num_nodes = len(self.map_dat['nodes'])

    
    def get_action(self, obs):
        # decide best action; random if sample < current epsilon
        actions = np.zeros(EVERGLADES_ACTION_SIZE)
        # Take best known actions
        # Get all swarm decisions
        swarm_decisions = self.get_all_swarm_decisions(obs)
        # Sort the swarm decisions array by the best q values
        sorted_swarm_decisions = sorted(
            swarm_decisions,
            key=lambda k: k['best_q_value']
        )
        actions[:] = [decision['best_action'] for decision in sorted_swarm_decisions[:7]]
        return actions

    def get_allies_on_node_data(self, obs):
        """
        Create a numpy array to store the number of allies on each node.

        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @returns A numpy array of size [NUM_NODES] with values of the number of ally swarms on each node
        """
        allies_on_node = np.zeros(self.num_nodes)
        for swarm_number in range(NUM_GROUPS):
            swarm_in_transit = obs[48+5*swarm_number]
            if swarm_in_transit == 0:
                swarm_node_loc = int(obs[45+5*swarm_number] - 1)
                allies_on_node[swarm_node_loc-1] += 1
        return allies_on_node

    def get_all_swarm_decisions(self, obs):
        """
        We want to obtain and return all swarm thought processes.
        
        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @returns An array of swarm thought processes, defined in the swarm_think documentation
        """
        # Create an array to store all swarm brain decisions / information
        swarm_decisions = []
        # We want to avoid creating the same computationally-expensive pre-processed
        # data multiple times, so we should calculate the number of ally nodes on each
        # node in the map here
        allies_on_node = self.get_allies_on_node_data(obs)
        # Iterate through all possible unit swarms, and have them each decide
        # which actions would be best for them to take
        for swarm_number in range(NUM_GROUPS):
            swarm_decisions.append(self.swarm_think(swarm_number, obs, allies_on_node))
        # Return the swarm decisions
        return swarm_decisions

    def swarm_think(self, swarm_number, obs, allies_on_node):
        """
        We want each individual swarm to think about which action it could take
        that would be best, and report back to the hivemind its thought processes.

        @param swarm_number The unit group number to have think
        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @param allies_on_node The number of allies on each node (nodes are 0-indexed)
        @returns swarm_thought_processes The total swarm thought processes object
                 swarm_thought_processes['best_action'] The best action the swarm found to take
                 swarm_thought_processes['best_q_value'] The best q value the swarm predicted
        """
        # Create the pre-processed observation space for the individual swarm
        swarm_obs = self.create_swarm_obs(swarm_number, obs, allies_on_node)
        # Find the predicted Q values for the swarm for all 12 possible actions
        with torch.no_grad():
            observation = torch.from_numpy(swarm_obs).float()
            swarm_predicted_q = self.policy_net(observation)
        # Find the best predicted node
        best_node = torch.argmax(swarm_predicted_q) + 1
        # Find the best predicted q value
        best_q_value = torch.max(swarm_predicted_q)
        # Create the swarm thought processes object to return
        swarm_thought_processes = {
            'best_action': np.array([swarm_number, best_node]),
            'best_q_value': best_q_value,
        }
        # Return the final swarm thought processes object
        return swarm_thought_processes

    
    def remember_game_state(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
        done=0,
    ):
        """
        Add the new experience to the NStepModule for experience replay.

        @param [previous_state] The previous observation state of the Everglades environment
        @param [next_state] The next observation state of the Everglades environment after taking specified action
        @param [actions] The actions that were taken to move agent from previous_state to next_state
        @param [reward] The reward received by the agent for taking specified actions in the previous_state
        """
        # -- We need to convert previous_state to the per-swarm's previous states --
        per_swarm_previous_state = np.zeros((NUM_GROUPS, INPUT_SIZE))
        # We can compute the number of allies on each node outside of the for loop
        previous_state_allies_on_node = self.get_allies_on_node_data(previous_state)
        for swarm_num in range(NUM_GROUPS):
            per_swarm_previous_state[swarm_num] = self.create_swarm_obs(swarm_num, previous_state, previous_state_allies_on_node)
            # -- We need to convert previous_state to the per-swarm's previous states --
        
        #print(per_swarm_previous_state.shape)
        #print(actions.shape)
        # Track the game in memory (the game itself is only integrated into the memory replay after the full game is played)
        if N_STEP > 1:
            self.NStepMemory.trackGameState(per_swarm_previous_state, actions, reward / 10000.0)

        self.ReplayMemory.trackGameState(per_swarm_previous_state, actions, reward / 10000.0)

    def create_swarm_obs(self, swarm_number, obs, allies_on_node):
        """
        Create the individual swarm's pre-processed input.

        @param swarm_number The unit group number to create the pre-processed input for
        @param obs The observation array consisting of all 105 values passed by the Everlgades environment
        @param allies_on_node The number of allies on each node (nodes are 0-indexed)
        @returns A numpy array of 47 values representing the swarm's observation
        """
        # Initialize an empty numpy array to store the swarm's observations
        swarm_obs = np.zeros(INPUT_SIZE)
        # Add the current turn number to the array
        swarm_obs[0] = obs[0] / 150.0
        # Add information on the control of each node on the map
        swarm_obs[1:12] = obs[3:45:4] / 100.0
        # Add information on the number of enemy units on each node on the map
        swarm_obs[12:23] = obs[4:45:4] / 100.0
        # Add information on the number of ally units on each node of the map
        swarm_obs[23:34] = allies_on_node[:] / 12.0
        # Add information on the swarm's current node
        for node_num in range(self.num_nodes):
            if obs[45+5*swarm_number] == node_num + 1:
                swarm_obs[34+node_num] = 1
            else:
                swarm_obs[34+node_num] = 0
        # Add information on the swarm's total health
        swarm_obs[45] = obs[47+5*swarm_number] * obs[49+5*swarm_number] / (1000.0)
        # Add information on whether the swarm is currently in transit
        swarm_obs[46] = obs[48+5*swarm_number]
        # Add swarm number ID to the input
        swarm_obs[47+swarm_number] = 1
        # Return the final swarm observation array
        return swarm_obs

    def optimize_model(self):
        """
        Optimize the network via a training function. Will return immediately
        without training if there is not enough memory in the experience replay.
        """
        self.update_count += 1
        if self.update_count % TARGET_UPDATE == 0:
            self.hard_update()

        if not self.ReplayMemory.isMemoryLargeEnoughToTrain(BATCH_SIZE):
            return

        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.ReplayMemory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"][np.argmax(weights)]
        # 1-step Learning loss
        elementwise_loss = self.compute_loss(samples, GAMMA, indices)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if N_STEP > 1:
            gamma = GAMMA ** N_STEP
            samples = self.NStepMemory.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self.compute_loss(samples, gamma, indices)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.ReplayMemory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()

    def compute_loss(self, samples: Dict[str, np.ndarray], gamma: float, indices) -> torch.Tensor:
        # """Return categorical dqn loss."""
        # device = self.device  # for shortening the following lines
        # state = torch.FloatTensor(samples["obs"]).to(device)
        # next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        # action = torch.LongTensor(samples["acts"]).to(device).unsqueeze(1)
        # reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        # done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        # next_state = state[0, :, :]
        
        # # Categorical DQN algorithm
        # delta_z = float(V_MAX - V_MIN) / (ATOM_SIZE - 1)

        # with torch.no_grad():
        #     # Double DQN
        #     # next_action = self.policy_net(next_state).gather(1, action)
        #     next_dist = torch.zeros((BATCH_SIZE, 12, self.num_nodes), device=device)
        #     for i in range(NUM_GROUPS):
        #         next_dist[0, i, :] = self.target_net(next_state[i, :]).detach()

        #     t_z = reward + (1 - done) * gamma * self.support
        #     t_z = t_z.clamp(min=V_MIN, max=V_MAX)
        #     b = (t_z - V_MIN) / delta_z
        #     l = b.floor().long()
        #     u = b.ceil().long()

        #     offset = (
        #         torch.linspace(
        #             0, (BATCH_SIZE - 1) * ATOM_SIZE, BATCH_SIZE
        #         ).long()
        #         .unsqueeze(1)
        #         .expand(BATCH_SIZE, ATOM_SIZE)
        #         .to(self.device)
        #     )

        #     proj_dist = torch.zeros(next_dist.size(), device=self.device)
        #     print(next_dist.shape)
        #     proj_dist.view(-1).index_add_(
        #         0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        #     )
        #     proj_dist.view(-1).index_add_(
        #         0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        #     )

        # dist = self.policy_net.dist(state).gather(1, action)
        # log_p = torch.log(dist[range(BATCH_SIZE), action])
        # elementwise_loss = -(proj_dist * log_p).sum(1)

        # return elementwise_loss
        nth_next_state_swarms_batch = torch.FloatTensor(samples["next_obs"]).to(device)
        nth_next_state_swarms_batch = nth_next_state_swarms_batch[indices]
        swarm_state_batch = torch.FloatTensor(samples["obs"]).to(device)
        print(swarm_state_batch.shape)
        swarm_state_batch = swarm_state_batch[indices]
        print(swarm_state_batch.shape)
        swarm_action_batch = torch.LongTensor(samples["acts"]).to(device).unsqueeze(1)
        swarm_action_batch = swarm_action_batch[indices]
        reward_batch = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        reward_batch = reward_batch[indices]
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        non_final_mask = non_final_mask[indices]
        non_final_next_state_swarms_batch = nth_next_state_swarms_batch[bool(non_final_mask[0]), :, :]
        
        # Compute the swarm's predicted qs for the current state
        print(swarm_state_batch.shape)
        print(swarm_action_batch.shape)
        print(self.policy_net(swarm_state_batch).shape)
        state_swarms_predicted_q_batch = self.policy_net(swarm_state_batch).gather(1, swarm_action_batch)

        # Compute the swarm's future value for next states
        next_state_swarms_predicted_qs_batch = torch.zeros((BATCH_SIZE, 12, self.num_nodes), device=device)
        for swarm_num in range(NUM_GROUPS):
            next_state_swarms_predicted_qs_batch[non_final_mask, swarm_num, :] = self.target_net(non_final_next_state_swarms_batch[:, swarm_num, :]).detach()
        # Limit future value to the best q value for each swarm
        max_next_state_swarms_predicted_qs_batch = torch.amax(next_state_swarms_predicted_qs_batch, axis=2)
        max_next_state_predicted_q_batch = torch.mean(max_next_state_swarms_predicted_qs_batch, axis=1)
        # Compute the estimated future reward
        estimated_future_reward = (max_next_state_predicted_q_batch * (GAMMA ** N_STEP) + reward_batch).to(device)

        # Compute the loss
        loss = F.smooth_l1_loss(state_swarms_predicted_q_batch, estimated_future_reward.type(torch.FloatTensor).unsqueeze(1))

        return loss

    def end_of_episode(self, episodes, total_episodes):
        """
        Perform end-of-episode functions for the agent such as updating the
        target network and saving game off to replay memory.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        # Update target network every UPDATE_TARGET_AFTER episodes
        if episodes % TARGET_UPDATE == 0 and TRAIN:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # Save the network every SAVE_NETWORK_AFTER episodes
        if episodes % SAVE_NETWORK_AFTER == 0 and TRAIN:
            self.save_network(episodes)
        # Decay epsilon
        if TRAIN:
            # noisy net deals with epsilon
            # beta for PER
            fraction = min(episodes / total_episodes, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

    def hard_update(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_network(self, episodes):
        """
        Saves the network's state dict, epsilon value, and episode count to the specified file.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        if NETWORK_SAVE_NAME:
            save_file = open(NETWORK_SAVE_NAME + '.pickle', 'wb')
            pickle.dump({
                'state_dict': self.policy_net.state_dict(),
                'epsilon': self.epsilon,
                'episodes': episodes + self.previous_episodes,
            }, save_file)
            save_file.close()
            print('Saved Network')
        else:
            print('Save Failed - Save File Not Specified')