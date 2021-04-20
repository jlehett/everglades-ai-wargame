import torch.nn as nn
import torch
from torch.distributions import Categorical
import gc

##########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    """
    Defines the Actor Critic Network to be used by RPPO
    """

    def __init__(self, state_dim, action_dim, n_latent_var, device, use_recurrent):
        """
        Setsup the Actor Critic class

        @param state_dim The size of the input state dimension
        @param action_dim The size of the output action dimension
        @param n_latent_var The number of nodes in each hidden layer (for both fully connected layers and GRU layers)
        @param device The device to load the tensors onto (will be cuda or cpu)
        @param use_recurrent Determines whether or not to use a recurrent architecture
        """
        super(ActorCritic, self).__init__()
        
        # Basic parameters
        self.num_layers = 2
        self.device = device
        self.n_latent_var = n_latent_var
        self.use_recurrent = use_recurrent
        
        # Actor layers
        self.action_head = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
        )

        self.action_gru = nn.GRU(n_latent_var, n_latent_var,batch_first=False)

        self.action_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh(),
            nn.Softmax(dim=-1)
        )

        # Critic layers
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
        )

        self.value_gru = nn.GRU(n_latent_var, n_latent_var, batch_first=False)

        self.value_layer = nn.Sequential(
            nn.Linear(n_latent_var, 1)
        )
    
    def forward(self):
        raise NotImplementedError

    def act(self, state = None, memory = None, hidden = None):
        """
        Queries the actor network to get an action to be used in everglades

        @param state The current state the agent is in
        @param memory The memory object to store action information in
        @param hidden The hidden state the agent is in (used for the GRU)
        @returns action_indices The indices of the chosen units and nodes for the current action
        @returns hidden The hidden state obtained from the GRU
        """
        # Query the actor
        state = torch.from_numpy(state).float().to(self.device)
        hidden = hidden.to(self.device)
        action_probs = self.action_head(state)

        # Use recurrent architecture
        if self.use_recurrent:
            # Convert action_probs to proper size for GRU
            action_probs = action_probs.unsqueeze(0).unsqueeze(0)

            # Repeat the obs for all seven actions to get from network
            action_probs = action_probs.repeat(7,1,1)

            action_probs,hidden = self.action_gru(action_probs, hidden)

        action_probs = self.action_layer(action_probs)

        # Resize final action_probs after using recurrent architecture
        if self.use_recurrent:
            # Get each action from each sequence output from the network
            action_indices = torch.empty(7).to(self.device)
            for i in range(7):
                dist = Categorical(action_probs[i][0])
                action_indices[i] = dist.sample()

                if not memory == None:
                    memory.logprobs.append(dist.log_prob(action_indices[i]))
                    memory.states.append(state)
                    memory.actions.append(action_indices[i])
        else:
            # Uses Boltzmann style exploration by sampling from distribution
            dist = Categorical(action_probs)

            # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
            # Enables us to grab non-duplicate actions faster
            action_indices = torch.multinomial(action_probs,7,replacement=False)

            # Append each action along with its log_prob and the current state separately
            # Makes the loss function more manageable
            if not memory == None:
                for i in range(7):
                    memory.logprobs.append(dist.log_prob(action_indices[i]))
                    memory.states.append(state)
                    memory.actions.append(action_indices[i])

        return action_indices, hidden

    def evaluate(self, state, action, hidden):
        """
        Queries the actor and critic networks for use in optimizing the model

        @param state The state batch stored in the agent memory
        @param action The action batch stored in the agent memory
        @param hidden The hidden batch (store or not stored in the memory)

        @returns action_logprobs The log of the action probs for use in ratio calculation in PPO
        @returns state_value The state values queried from the critic
        @returns dist_entropy The calculated entropy from the actor
        """

        # Query the actor
        action_probs = self.action_head(state)

        # Use recurrent architecture
        if self.use_recurrent:
            # Convert action_probs to proper size for GRU
            action_probs = action_probs.unsqueeze(0)

            # Repeat the obs for all seven actions to get from network
            action_probs = torch.reshape(action_probs, (7,action_probs.size(1) // 7,self.n_latent_var))

            action_probs,hidden = self.action_gru(action_probs, hidden)
        
        action_probs = self.action_layer(action_probs)

        # Resize action_probs after using recurrent architecture
        if self.use_recurrent:
            # Use same distribution as act
            dist = Categorical(action_probs)

            # Calculate the expected log_probs for the previous actions
            action = action.reshape(7, action.size(0) // 7)
            action_logprobs = dist.log_prob(action)
            action_logprobs = action_logprobs.reshape(action_logprobs.size(1) * 7)

            # Calculate the entropy from the distribution
            dist_entropy = dist.entropy()
            dist_entropy = dist_entropy.reshape(dist_entropy.size(1) * 7)
        else:
            # Use same distribution as act
            dist = Categorical(action_probs)

            # Calculate the expected log_probs for the previous actions
            action_logprobs = dist.log_prob(action)

            # Calculate the entropy from the distribution
            dist_entropy = dist.entropy()

        # Query the critic / Get expected network output
        state_value = self.value_head(state)

        # Use recurrent architecture
        if self.use_recurrent:
            # Convert state_value to proper size for GRU
            state_value = state_value.unsqueeze(0)

            # Repeat the obs for all seven actions to get from network
            state_value = torch.reshape(state_value, (7,state_value.size(1) // 7,self.n_latent_var))

            state_value,_ = self.value_gru(state_value,hidden)

        state_value = self.value_layer(state_value)

        # Resize state_value after using recurrent architecture
        if self.use_recurrent:
            state_value = state_value.reshape(state_value.size(1) * 7)

        return action_logprobs, state_value, dist_entropy