from agents.PPO.ActorCritic import ActorCritic

import torch
import torch.optim as optim
import torch.nn.functional as F
import json
import random
import numpy as np
from collections import namedtuple
import pickle
import os

INPUT_SIZE = 105
OUTPUT_SIZE = 132

class PPO():
    """
    PPO Class for evaluation post training
    """
    def __init__(self, save_file_data, player_num, map_name):
        """
        Initialize the PPO evaluation class

        @param save_file_data The save file containing the trained PPO's data
        @param player_num The player number of the agent
        @param map_name The name of the everglades map
        """
        # Store the save file data
        self.save_file_data = save_file_data

        # Load the saved policy net
        self.policy = ActorCritic(INPUT_SIZE, OUTPUT_SIZE, save_file_data['n_latent_var'], torch.device('cpu'), save_file_data['use_recurrent'])
        self.policy.load_state_dict(save_file_data['policy_state_dict'])

        # Final action shape
        self.shape = (7,2)

        # Set up the hidden states for recurrent
        self.hidden = torch.zeros(self.save_file_data['n_latent_var']).unsqueeze(0).unsqueeze(0)

    def get_action(self, obs):
        """
        Gets an agent action at a particular time step

        @param observation: The observation of the agent in the current turn
        @param hidden: The previous hidden state of the agent's GRU
        """
        action = np.zeros(self.shape)
        chosen_indices, self.hidden = self.policy.act(state=obs, hidden=self.hidden)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units.cpu()
        action[:,1] = chosen_nodes.cpu()
        return action