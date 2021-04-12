from agents.Smart_State.QNetwork import QNetwork
from agents.Smart_State.Multi_Step import NStepModule
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import random
import numpy as np
from collections import namedtuple
import pickle
import os

from agents.Smart_State.Move_Translation import get_move

EVALUATE_EPSILON = 0.0 # The epsilon value to use when evaluating the network (when TRAIN is set to False)
TRAIN_EPSILON_START = 0.95 # The epsilon value to use when starting to train the network (when TRAIN is set to True)
TRAIN_EPSILON_MIN = 0.05 # The minimum epsilon value to use during training (when TRAIN is set to True)
TRAIN_LR_START = 1e-5 # The learning rate value to use when starting to train the network (when TRAIN is set to True)
TRAIN_LR_MIN = 1e-8 # The minimum learning rate value to use during training (when TRAIN is set to True)

SAVE_NETWORK_AFTER = 10 # Save the network every n episodes

NUM_GROUPS = 12 # The number of unit groups in the Everglades environment for the agent
NUM_NODES = 11 # The number of nodes in the Everglades environment
NUM_ACTIONS = 7 # The number of actions an agent can take in a single turn
EVERGLADES_ACTION_SIZE = (NUM_ACTIONS, 2) # The action shape in an Everglades-readable format

INPUT_SIZE = 59 # This is a custom value defined when creating the smart state agent
OUTPUT_SIZE = 5 # This is a custom value defined when creating the smart state agent
FC1_SIZE = 80 # Number of nodes in the first hidden layer
FC2_SIZE = 80 # number of nodes in the second hidden layer

BATCH_SIZE = 5012 # The number of inputs to train on at one time
TARGET_UPDATE = 500 # The number of episodes to wait until we update the target network
MEMORY_SIZE = 100000 # The number of experiences to store in memory replay
GAMMA = 0.999 # The amount to discount the future rewards by
N_STEP = 1 # The number of steps to use in multi-step learning
EPS_DECAY = 0.995 # The rate at which epsilon decays at the end of each episode
LR_DECAY = 0.999 # The rate at which epsilon decays at the end of each episode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQNAgent():
    def __init__(
        self,
        player_num,
        map_name,
        train=True,
        network_save_name=None,
        network_load_name=None,
    ):
        """
        Initialize a DQNAgent that will be used to play the Everglades game.
        Initializes class variables, creates policy and target networks, and
        loads the map data.

        @param player_num The player number of the agent in the Everglades environment
        @param map_name The name of the map file to load in
        """
        # Store variables
        self.network_save_name = network_save_name
        self.network_load_name = network_load_name
        self.fc1_size = FC1_SIZE
        self.fc2_size = FC2_SIZE
        self.learning_rate = TRAIN_LR_START
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE
        self.memory_size = MEMORY_SIZE
        self.n_step = N_STEP
        self.gamma = GAMMA
        self.train = train
        if self.train:
            self.epsilon = TRAIN_EPSILON_START
        else:
            self.epsilon = EVALUATE_EPSILON
        self.previous_episodes = 0
        self.training = self.train

        self.loss = 0

        # Load save information if it exists
        save_file_data = None
        if self.network_load_name and os.path.exists(self.network_load_name + '.pickle'):
            save_file = open(self.network_load_name + '.pickle', 'rb')
            save_file_data = pickle.load(save_file)
            save_file.close()

        if save_file_data:
            self.fc1_size = save_file_data.get('fc1_size')
            self.fc2_size = save_file_data.get('fc2_size')
            self.batch_size = save_file_data.get('batch_size')
            self.n_step = save_file_data.get('n_step')
            self.gamma = save_file_data.get('gamma')
            self.memory_size = save_file_data.get('memory_size')

            if self.train:
                self.epsilon = save_file_data.get('epsilon')
                self.previous_episodes = save_file_data.get('episodes')

        # Create the NStepModule
        self.NStepModule = NStepModule(self.n_step, self.gamma, self.memory_size)

        # Set up the network
        self.policy_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, self.fc1_size, self.fc2_size)
        self.target_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, self.fc1_size, self.fc2_size)
        
        self.policy_net.cuda()
        self.target_net.cuda()
        
        

        # Load up the save policy network data if it exists
        if save_file_data:
            self.policy_net.load_state_dict(save_file_data.get('policy_state_dict'))
            self.target_net.load_state_dict(save_file_data.get('target_state_dict'))
            print('Loaded Saved Network:', self.network_load_name)
        else:
            print('No Network found for player:', player_num)
            # Otherwise, we need to copy the policy network's weights to the target network to
            # begin training
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Prevent the target net from learning during training; only the policy
        # net should be learning
        self.target_net.eval()

        # Set up the map data to use
        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)
            self.nodes_map = {}
            for i, in_node in enumerate(self.map_dat['nodes']):
                self.nodes_map[in_node['ID']] = in_node
            self.num_nodes = len(self.map_dat['nodes'])

    def get_action(self, obs):
        """
        Get an action for the current step in the environment.
        If a random sample between [0, 1] < epsilon, use a random
        action. Otherwise, exploit the best known action based on network
        predictions.

        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @returns The 7x2 tuple containing the final actions the agent has opted to take in an Everglades-readable format
        """
        # Get the random sample to determine whether to use random actions
        # or the best known actions.
        sample = random.random()
        if sample < self.epsilon:
            # Take random actions
            return self.get_random_actions(obs)
        else:
            # Take best known actions
            return self.get_best_actions(obs)

    def get_random_actions(self, obs):
        """
        Get 7 random actions the agent should take in the Everglades environment.

        @returns 7x2 Numpy array tuple containing the final actions the agent has opted to take in an Everglades-readable format
        """
        # Create empty array for action storage
        actions = np.zeros(NUM_ACTIONS)
        # Determine the unit numbers randomly
        swarms = np.random.choice(NUM_GROUPS, NUM_ACTIONS, replace=False)
        # Determine the directions randomly
        directions = np.random.choice(OUTPUT_SIZE, NUM_ACTIONS, replace=True)
        # Grab each swarm's action
        for i, swarm_num in enumerate(swarms):
            swarm_node_number_0_indexed = self.get_swarm_node_number(swarm_num, obs)
            action = get_move(swarm_node_number_0_indexed, directions[i])
            actions[i] = action
        # Create the final action selections array
        final_action_array = np.zeros(EVERGLADES_ACTION_SIZE)
        final_direction_array = np.zeros(EVERGLADES_ACTION_SIZE)
        for i in range(NUM_ACTIONS):
            final_action_array[i] = np.array([swarms[i], actions[i]])
            final_direction_array[i] = np.array([swarms[i], directions[i]])
        # Return both the final action array and the selected directions
        return final_action_array, final_direction_array

    def get_best_actions(self, obs):
        """
        Get the 7 actions the agent believes are the best options given the current
        observation. Each unit swarm the agent controls should come up with their own
        Q values and best action to take. This function will take the 7 best actions
        that are returned among all unit swarms.

        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @returns 7x2 Numpy array containing the best actions the agent has predicted
        """
        # Get all swarm decisions
        swarm_decisions = self.get_all_swarm_decisions(obs)
        # Sort the swarm decisions array by the best q values
        sorted_swarm_decisions = sorted(
            swarm_decisions,
            key=lambda k: k['best_q_value']
        )
        # Return the top 7 actions
        actions = np.zeros(EVERGLADES_ACTION_SIZE)
        directions = np.zeros(EVERGLADES_ACTION_SIZE)
        actions[:] = [decision['best_action'] for decision in sorted_swarm_decisions[:7]]
        directions[:] = [decision['best_direction'] for decision in sorted_swarm_decisions[:7]]
        return actions, directions

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
                allies_on_node[swarm_node_loc] += 1
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
        swarm_decisions
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
                swarm_predicted_q = self.policy_net(swarm_obs)
        # Find the best predicted direction
        best_direction = torch.argmax(swarm_predicted_q).to(device)
        # Use the Move_Translation helper file to get the move the swarm would take by going in that direction
        swarm_node = self.get_swarm_node_number(swarm_number, obs)
        best_node = get_move(swarm_node, best_direction)
        # Find the best predicted q value
        best_q_value = torch.max(swarm_predicted_q).to(device)
        # Create the swarm thought processes object to return
        swarm_thought_processes = {
            'best_direction': np.array([swarm_number, best_direction]),
            'best_action': np.array([swarm_number, best_node]),
            'best_q_value': best_q_value,
        }
        # Return the final swarm thought processes object
        return swarm_thought_processes

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
        swarm_obs = torch.from_numpy(swarm_obs).to(device)

        return swarm_obs

    def get_swarm_node_number(self, swarm_number, obs):
        """
        Grab the node number that a swarm is at given the observation and swarm number.

        @param swarm_number The swarm to query
        @param ibs The observation array consisting of all 105 values passed by the Everglades environment
        @returns An integer representing the node number the swarm is at in a 0-indexed system
        """
        return obs[45+5*swarm_number] - 1

    def remember_game_state(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
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
            per_swarm_previous_state[swarm_num] = self.create_swarm_obs(swarm_num, previous_state, previous_state_allies_on_node).cpu()
        # Track the game in memory (the game itself is only integrated into the memory replay after the full game is played)
        self.NStepModule.trackGameState(per_swarm_previous_state, actions, reward)

    def optimize_model(self):
        """
        Optimize the network via a training function. Will return immediately
        without training if there is not enough memory in the experience replay.
        """
        # If training is not set to true, we do not want to optimize the model
        if not self.train:
            return
        # If the NStepModule's experience replay isn't large enough, we should bail out.
        # Otherwise, we can grab sample data from the replay memory.
        if not self.NStepModule.isMemoryLargeEnoughToTrain(self.batch_size):
            return
        transitions = self.NStepModule.sampleReplayMemory(self.batch_size)

        # Set the optimizer to use in training the network using the latest learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Create the batch of data to use
        batch = Transition(*zip(*transitions))
        nth_next_state_swarms_batch = torch.from_numpy(np.asarray(batch.next_state_swarms)).to(device)
        swarm_state_batch = torch.from_numpy(np.asarray(batch.swarm_obs)).to(device)
        swarm_action_batch = torch.from_numpy(np.asarray(batch.swarm_action)).unsqueeze(1).to(device)
        reward_batch = torch.from_numpy(np.asarray(batch.reward)).to(device)
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.from_numpy(np.asarray(batch.doesNotHitDone)).to(device)
        non_final_next_state_swarms_batch = nth_next_state_swarms_batch[non_final_mask, :, :]

        # Compute the swarm's predicted qs for the current state
        state_swarms_predicted_q_batch = self.policy_net(swarm_state_batch).gather(1, swarm_action_batch).to(device)

        # Compute the swarm's future value for next states
        next_state_swarms_predicted_qs_batch = torch.zeros((self.batch_size, 12, OUTPUT_SIZE), device=device).to(device)
        for swarm_num in range(NUM_GROUPS):
            next_state_swarms_predicted_qs_batch[non_final_mask, swarm_num, :] = self.target_net(non_final_next_state_swarms_batch[:, swarm_num, :]).detach()
        # Limit future value to the best q value for each swarm
        max_next_state_swarms_predicted_qs_batch = torch.amax(next_state_swarms_predicted_qs_batch, axis=2).to(device)
        max_next_state_predicted_q_batch = torch.mean(max_next_state_swarms_predicted_qs_batch, axis=1).to(device)
        # Compute the estimated future reward
        estimated_future_reward = (max_next_state_predicted_q_batch * (self.gamma ** self.n_step) + reward_batch).to(device)

        # Compute the loss
        loss = F.smooth_l1_loss(state_swarms_predicted_q_batch.type(torch.FloatTensor).to(device), estimated_future_reward.type(torch.FloatTensor).unsqueeze(1).to(device)).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Sets the loss to be grabbed by training file
        self.loss = loss.detach()

    def end_of_episode(self, episodes):
        """
        Perform end-of-episode functions for the agent such as updating the
        target network and saving game off to replay memory.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        # Add the played game to memory
        self.NStepModule.addGameToReplayMemory()
        # Call the rest of the end of episode logic
        self.end_of_episode_not_play(episodes)

    def end_of_episode_not_play(self, episodes):
        """
        Performs the end of episode updates without the need of having played this episode.
        To be used in self-play tournaments.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        # Optimize the model
        self.optimize_model()
        # Update target network every UPDATE_TARGET_AFTER episodes
        if (episodes + self.previous_episodes) % self.target_update == 0 and self.train:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # Decay epsilon
        if self.train:
            self.epsilon *= EPS_DECAY
            if self.epsilon < TRAIN_EPSILON_MIN:
                self.epsilon = TRAIN_EPSILON_MIN
        # Decay learning rate
        if self.train:
            self.learning_rate *= LR_DECAY
            if self.learning_rate < TRAIN_LR_MIN:
                self.learning_rate = TRAIN_LR_MIN

    def save_network(self, episodes):
        """
        Saves the network's state dict, epsilon value, and episode count to the specified file.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        if self.network_save_name:
            save_file = open(os.getcwd() + self.network_save_name + '.pickle', 'wb')
            pickle.dump({
                'policy_state_dict': self.policy_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'epsilon': self.epsilon,
                'episodes': episodes + self.previous_episodes,
                'fc1_size': self.fc1_size,
                'fc2_size': self.fc2_size,
                'batch_size': self.batch_size,
                'target_update': self.target_update,
                'memory_size': self.memory_size,
                'gamma': self.gamma,
                'n_step': self.n_step,
            }, save_file)
            save_file.close()
            print('Saved Network')
        else:
            print('Save Failed - Save File Not Specified')


### DEFINE REPLAY MEMORY TRANSITION ###

Transition = namedtuple('Transition',
                        ('swarm_obs', 'swarm_action', 'next_state_swarms', 'reward', 'doesNotHitDone'))
