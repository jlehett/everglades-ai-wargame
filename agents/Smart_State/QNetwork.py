import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc1_unit):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(1)

        # Define the network
        self.fc1 = nn.Linear(input_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, output_size)

    def forward(self, x):
        # Convert the input to a tensor if it was a numpy array
        if (type(x).__module__ == np.__name__):
            x = torch.from_numpy(x).to(device)

        # Perform forward pass
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Return the final output
        return x