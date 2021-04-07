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

NUM_GROUPS = 12 # The number of unit groups in the Everglades environment for the agent
NUM_NODES = 11 # The number of nodes in the Everglades environment
NUM_ACTIONS = 7 # The number of actions an agent can take in a single turn
EVERGLADES_ACTION_SIZE = (NUM_ACTIONS, 2) # The action shape in an Everglades-readable format

INPUT_SIZE = 59 # This is a custom value defined when creating the smart state agent
OUTPUT_SIZE = 5 # This is a custom value defined when creating the smart state agent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
AGENT CLASS
"""

class SmartStateFixedFC2():
    def __init__(
        self,
        save_file_data,
        player_num,
        map_name
    ):
        # Store the save file data
        self.save_file_data = save_file_data

        # Load the saved policy net
        self.policy_net = QNetwork(INPUT_SIZE, OUTPUT_SIZE, save_file_data['fc1_size'], save_file_data['fc2_size'])
        self.policy_net.load_state_dict(save_file_data['policy_state_dict'])

    def get_action(self, obs):
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
        actions[:] = [decision['best_action'] for decision in sorted_swarm_decisions[:7]]
        return actions

    def get_allies_on_node_data(self, obs):
        """
        Create a numpy array to store the number of allies on each node.
        
        @param obs The observation array consisting of all 105 values passed by the Everglades environment
        @returns A numpy array of size [NUM_NODES] with values of the number of ally swarms on each node
        """
        allies_on_node = np.zeros(NUM_NODES)
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
        best_direction = torch.argmax(swarm_predicted_q)
        # Use the Move_Translation helper file to get the move the swarm would take by going in that direction
        swarm_node = self.get_swarm_node_number(swarm_number, obs)
        best_node = get_move(swarm_node, best_direction)
        # Find the best predicted q value
        best_q_value = torch.max(swarm_predicted_q)
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
        for node_num in range(NUM_NODES):
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

    def get_swarm_node_number(self, swarm_number, obs):
        """
        Grab the node number that a swarm is at given the observation and swarm number.
        
        @param swarm_number The swarm to query
        @param ibs The observation array consisting of all 105 values passed by the Everglades environment
        @returns An integer representing the node number the swarm is at in a 0-indexed system
        """
        return obs[45+5*swarm_number] - 1


"""
Q-NETWORK CLASS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc1_unit, fc2_unit):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(1)

        # Define the network
        self.fc1 = nn.Linear(input_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, output_size)

    def forward(self, x):
        # Convert the input to a tensor if it was a numpy array
        if (type(x).__module__ == np.__name__):
            x = torch.from_numpy(x)

        # Perform forward pass
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Return the final output
        return x


"""
HELPER TO TRANSLATE CARDINAL DIRECTIONS TO ACTUAL MOVES
"""

move_translator_left = {
    1: 1,
    2: 1,
    3: 3,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 11
}

move_translator_right = {
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 9,
    7: 10,
    8: 11,
    9: 9,
    10: 11,
    11: 11
}

move_translator_up = {
    1: 2,
    2: 2,
    3: 2,
    4: 3,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 8,
    10: 9,
    11: 8
}

move_translator_down = {
    1: 4,
    2: 3,
    3: 4,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 9,
    9: 10,
    10: 10,
    11: 10
}

move_translator_stay = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11
}

move_translator_total = {
    0: move_translator_left,
    1: move_translator_right,
    2: move_translator_up,
    3: move_translator_down,
    4: move_translator_stay,
}

# Helper function to determine where a swarm should move given
# its current location and the direction it should move to
#
# @param 
def get_move(node_location_0_indexed, direction):
    """
    Helper function to determine where a swarm should move given
    its current location and the direction it should move.
    
    @param node_location_0_indexed The location the swarm is currently
        at using a 0-indexed system
    @param direction The direction the swarm should move as an integer
        between 0 and 3, each representing a different cardinal direction
    @returns The node the swarm should move to using a 1-indexed
        system
    """
    return move_translator_total[int(direction)][int(node_location_0_indexed) + 1]
