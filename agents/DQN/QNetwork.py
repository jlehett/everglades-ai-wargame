import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T

from agents.DQN.NoisyLinear import NoisyLinear

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,observation_size,action_size, seed, fc1_unit = 528,
                 fc2_unit = 528):
        """
        Initialize parameters and build model.
        Params
        =======
            observation_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        self.action_size = 12 * 11
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(observation_size[0],fc1_unit)
        
        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################
        self.fc2 = nn.Linear(fc1_unit,self.action_size)

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################

        #self.fc2_adv = nn.Linear(fc1_unit,fc2_unit)
        #self.fc2_val = nn.Linear(fc1_unit,fc2_unit)

        #self.fc3_adv = nn.Linear(fc2_unit,self.action_size)
        #self.fc3_val = nn.Linear(fc2_unit,1)

        ##############################################################################################
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        #print("x: ", x)
        #print("x-type", type(x))
        if(type(x).__module__ == np.__name__):
            x = torch.from_numpy(x).float()
        
        x = x.float()
        
        x = F.relu(self.fc1(x))
        
        #############################################################################################
        #   Non-Dueling Architecture                                                                #
        #############################################################################################

        x = F.relu(self.fc2(x))

        #############################################################################################

        #############################################################################################
        # Dueling Network Architecture                                                              #
        # Code based on dxyang DQN agent https://github.com/dxyang/DQN_pytorch/blob/master/model.py #
        #############################################################################################
        
        #adv = F.relu(self.fc2_adv(x))
        #val = F.relu(self.fc2_val(x))

        #adv = self.fc3_adv(adv)
        #val = self.fc3_val(val)

        #advAverage = adv.mean()
        #x = val + adv - advAverage

       ##############################################################################################

        return x
   
    def reset_noise(self):
        """
        Reset the noise parameters for Noisy Linear
        """
        self.fc1.reset_noise()
        #self.fc2.reset_noise() # Non-dueling

        self.fc2_adv.reset_noise()
        self.fc2_val.reset_noise()
        self.fc3_adv.reset_noise()
        self.fc3_val.reset_noise()