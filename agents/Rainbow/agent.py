
import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import random
import numpy as np
from collections import namedtuple
import pickle
import os

from agent_attributes.PER import PrioritizedReplayBuffer
from agent_attributes.Network import Network
from ReplayMemory import ReplayBuffer

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
N_STEP = 1 # The number of steps to use in multi-step learning
EPS_DECAY = 0.999 # The rate at which epsilon decays at the end of each episode

# distributional
V_MIN = 0.0
V_MAX = 200.0

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
        if TRAIN:
            self.epsilon = TRAIN_EPSILON_START
        else:
            self.epsilon = EVALUATE_EPSILON
        self.previous_episodes = 0
        self.training = TRAIN

        # Create the NStepModule
        #self.NStepModule = NStepModule(N_STEP, GAMMA, MEMORY_SIZE)
        if N_STEP > 1:
            self.NStepMemory = ReplayBuffer(observation_space, MEMORY_SIZE, BATCH_SIZE, N_STEP, GAMMA)

        # PER memory
        self.ReplayMemory = PrioritizedReplayBuffer(observation_space, MEMORY_SIZE, BATCH_SIZE, ALPHA)

        # Set up the network
        #self.policy_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE)
        #self.target_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE)
        self.support = torch.linspace(V_MIN, V_MAX, FC1_SIZE).to(device)
        self.policy_net = Network(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE, self.support)
        self.target_net = Network(INPUT_SIZE, OUTPUT_SIZE, FC1_SIZE, self.support)

        # If a save file is specified and the file exists, load the save file
        if NETWORK_LOAD_NAME and os.path.exists(NETWORK_LOAD_NAME + '.pickle'):
            save_file = open(NETWORK_LOAD_NAME + '.pickle', 'rb')
            save_file_data = pickle.load(save_file)
            self.policy_net.load_state_dict(save_file_data.get('state_dict'))
            if TRAIN:
                self.epsilon = save_file_data.get('epsilon')
                self.previous_episodes = save_file_data.get('episodes')
            save_file.close()
            print('Loaded Saved Network:', NETWORK_LOAD_NAME)

        # Load the policy network's values into the target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Prevent the target net from learning during training; only the policy
        # net should be learning
        self.target_net.eval()

        # Set the optimizer to use in training the network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

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
            return self.get_random_actions()
        else:
            # Take best known actions
            return self.get_best_actions(obs)

    def get_random_actions(self):
        """
        Get 7 random actions the agent should take in the Everglades environment.

        @returns 7x2 Numpy array tuple containing the final actions the agent has opted to take in an Everglades-readable format
        """
        # Create empty array for action storage
        actions = np.zeros(EVERGLADES_ACTION_SIZE)
        # Determine the unit numbers randomly
        actions[:, 0] = np.random.choice(NUM_GROUPS, NUM_ACTIONS, replace=False)
        actions[:, 1] = np.random.choice(self.num_nodes, NUM_ACTIONS, replace=False) + 1
        # Return the final action selections
        return actions

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
        # for decision in sorted_swarm_decisions:
        #     print('Swarm: ' + str(decision['best_action'][0]) + '\t| Node: ' + str(decision['best_action'][1]) + '\t| Q-value: ' + str(decision['best_q_value'].item()))
        # print('')
        # Return the top 7 actions
        actions = np.zeros(EVERGLADES_ACTION_SIZE)
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
            swarm_predicted_q = self.policy_net(swarm_obs)
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
            per_swarm_previous_state[swarm_num] = self.create_swarm_obs(swarm_num, previous_state, previous_state_allies_on_node)
        # Track the game in memory (the game itself is only integrated into the memory replay after the full game is played)
        self.ReplayMemory.trackGameState(per_swarm_previous_state, actions, reward / 10000.0)

    def optimize_model(self):
        """
        Optimize the network via a training function. Will return immediately
        without training if there is not enough memory in the experience replay.
        """
        # If training is not set to true, we do not want to optimize the model
        if not TRAIN:
            return
        # If the NStepModule's experience replay isn't large enough, we should bail out.
        # Otherwise, we can grab sample data from the replay memory.
        if not self.ReplayMemory.isMemoryLargeEnoughToTrain(BATCH_SIZE):
            return
        transitions = self.ReplayMemory.sampleReplayMemory(BATCH_SIZE)

        # Create the batch of data to use
        batch = Transition(*zip(*transitions))
        nth_next_state_swarms_batch = torch.from_numpy(np.asarray(batch.next_state_swarms))
        swarm_state_batch = torch.from_numpy(np.asarray(batch.swarm_obs))
        swarm_action_batch = torch.from_numpy(np.asarray(batch.swarm_action)).unsqueeze(1)
        reward_batch = torch.from_numpy(np.asarray(batch.reward))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.from_numpy(np.asarray(batch.doesNotHitDone))
        non_final_next_state_swarms_batch = nth_next_state_swarms_batch[non_final_mask, :, :]

        # Compute the swarm's predicted qs for the current state
        state_swarms_predicted_q_batch = self.policy_net(swarm_state_batch).gather(1, swarm_action_batch)

        # Compute the swarm's future value for next states
        next_state_swarms_predicted_qs_batch = torch.zeros((BATCH_SIZE, 12, self.num_nodes), device=device)
        next_action_taken_mask = torch.zeros((BATCH_SIZE, 12, 1), device=device).type(torch.int64)
        for swarm_num in range(NUM_GROUPS):
            # Use the policy net to get the action it would have taken for the swarm in the next state
            next_action_taken_mask[:, swarm_num, 0] = torch.argmax(self.policy_net(nth_next_state_swarms_batch[:, swarm_num, :]), axis=1).type(torch.int64)
            # Use the target net to compute all possible action values for next state.
            next_state_swarms_predicted_qs_batch[non_final_mask, swarm_num, :] = self.target_net(non_final_next_state_swarms_batch[:, swarm_num, :]).detach()
        # Limit future value to the avg q value of the actions that would have been chosen for each swarm
        #print(next_action_taken_mask)
        selected_next_state_swarms_predicted_qs_batch = next_state_swarms_predicted_qs_batch.gather(2, next_action_taken_mask)
        max_next_state_predicted_q_batch = torch.mean(selected_next_state_swarms_predicted_qs_batch, axis=1)
        # Compute the estimated future reward
        estimated_future_reward = (max_next_state_predicted_q_batch * (GAMMA ** N_STEP) + reward_batch.unsqueeze(1)).to(device)

        # Compute the loss
        loss = F.smooth_l1_loss(state_swarms_predicted_q_batch, estimated_future_reward.type(torch.FloatTensor))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def end_of_episode(self, episodes):
        """
        Perform end-of-episode functions for the agent such as updating the
        target network and saving game off to replay memory.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        # Add the played game to memory
        self.ReplayMemory.addGameToReplayMemory()
        # Update target network every UPDATE_TARGET_AFTER episodes
        if episodes % TARGET_UPDATE == 0 and TRAIN:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # Save the network every SAVE_NETWORK_AFTER episodes
        if episodes % SAVE_NETWORK_AFTER == 0 and TRAIN:
            self.save_network(episodes)
        # Decay epsilon
        if TRAIN:
            self.epsilon *= EPS_DECAY
            if self.epsilon < TRAIN_EPSILON_MIN:
                self.epsilon = TRAIN_EPSILON_MIN

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

Transition = namedtuple('Transition',
                        ('swarm_obs', 'swarm_action', 'next_state_swarms', 'reward', 'doesNotHitDone'))