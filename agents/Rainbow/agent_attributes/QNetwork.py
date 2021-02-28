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
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc1_unit, fc2_unit):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(1)

        # Define the network
        self.fc1 = nn.Linear(input_size, fc1_unit)
        
        self.fc3_adv = nn.Linear(fc1_unit,fc2_unit)
        self.fc3_val = nn.Linear(fc1_unit,fc2_unit)

        self.fc4_adv = nn.Linear(fc2_unit,self.output_size)
        self.fc4_val = nn.Linear(fc2_unit,1)

    def forward(self, x):
        # Convert the input to a tensor if it was a numpy array
        if (type(x).__module__ == np.__name__):
            x = torch.from_numpy(x)

        # Perform forward pass
        x = x.float()
        x = F.relu(self.fc1(x))
        
        adv = F.relu(self.fc3_adv(x))
        val = F.relu(self.fc3_val(x))

        adv = self.fc4_adv(adv)
        val = self.fc4_val(val)

        advAverage = adv.mean()
        x = val + adv - advAverage

        # Return the final output
        return x
