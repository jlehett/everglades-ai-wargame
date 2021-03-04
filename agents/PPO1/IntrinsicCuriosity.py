import torch
import numpy as np
import torch.nn as nn

class ICM(nn.Module):
    # Add swish activation
    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        self.act_dim = action_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh()
        )

        # Inverse model
        self.inverse = nn.Sequential(
            nn.Linear(n_latent_var * 2, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,action_dim)
        )

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(n_latent_var + action_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var)
        )

        # Loss Functions
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')

    def forward(self, act, curr_obs, next_obs, mask):
        # Inverse model
        curr_enc = self.encoder(curr_obs)
        next_enc = self.encoder(next_obs)
        combined = torch.cat([curr_enc,next_enc], dim=1)
        pred_act = self.inverse(combined)#torch.transpose(self.inverse(combined), 0, 1)
        inv_loss = (self.CrossEntropyLoss(pred_act, act.long()) * mask).mean()

        # Forward model
        one_hot_act = nn.functional.one_hot(act, num_classes=self.act_dim)
        pred_next_enc = self.forward_model(torch.cat((one_hot_act.float(), curr_enc), dim=1))

        # Intrinsic reward
        intr_reward = 0.5 * self.MSELoss(pred_next_enc, next_enc)
        intr_reward = intr_reward.mean(dim=1) * mask

        # Forward loss
        forw_loss = intr_reward.mean()
        return intr_reward, inv_loss, forw_loss



class Swish(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return self.swish(input) # simply apply already implemented SiLU

    def swish(self, input):
        return input * torch.sigmoid(input)