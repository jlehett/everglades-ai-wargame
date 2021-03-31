import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym

global device 
device = torch.device('cpu')

#######################
#   PPO Agent Class   #
#######################

class RPPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_latent_var,
        lr,
        betas,
        gamma,
        K_epochs, 
        eps_clip,
        lambd,
        DEVICE
    ):
        # Enables GPU Training
        global device
        if DEVICE == 'GPU':
            device = torch.device('cuda')


        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.n_latent_var = n_latent_var

        self.memory = Memory()

        # Setup policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        # Do not use RMSProp (learned that the hard way). Adam is the optimizer to use for PPO
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.shape = (7, 2)
        self.loss = 0
        self.actor_loss = 0
        self.critic_loss = 0
        self.dist_entropy = 0

        # Set the loss function
        # Only use MSELoss for PPO
        self.MSE = nn.MSELoss()

    def get_action(self, observation, hidden):
        action = np.zeros(self.shape)
        chosen_indices, hidden = self.policy_old.act(observation,self.memory, hidden)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units.cpu()
        action[:,1] = chosen_nodes.cpu()
        return action, hidden

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        gae = 0

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        next_states = torch.stack(self.memory.next_states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        mask = torch.stack(self.memory.is_terminals).to(device).detach()
        reward = torch.from_numpy(np.asarray(self.memory.rewards)).to(device).detach()
        
        hidden = torch.zeros(old_states.size(0),self.n_latent_var).unsqueeze(0).to(device)

        _,values,_ = self.policy.evaluate(old_states, old_actions, hidden)
        values = values.detach()
        last_value = values[len(values) - 1]

        # Calculate the Advantage
        for i in reversed(range(len(self.memory.rewards))):
            ### Generalized Advantage Estimation ###
            delta = (reward[i]) + (self.gamma * last_value * mask[i]) - values[i]
            gae = delta + (self.gamma * self.lambd * mask[i] * gae)
            last_value = values[i]
            rewards.insert(0, gae + values[i])
            ### End Generalized Advantage Estimation ###

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device) - values[:]
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        stop_early = False
        # Optimize policy for K epochs:
        for k in range(self.K_epochs):

            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, hidden)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding surrogate loss:
            advantages = rewards
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            actor_loss = torch.min(surr1, surr2)
            critic_loss = self.MSE(state_values, rewards) * 0.5
            entropy = 0.01*dist_entropy
            loss = -actor_loss + critic_loss - entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.loss = loss.detach()
            self.actor_loss = actor_loss.mean().detach()
            self.critic_loss = critic_loss.detach()
            self.dist_entropy = entropy.mean().detach()

            k_epochs = k
            if stop_early:
                break

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

####################
#   Memory Class   #
####################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

##########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.num_layers = 2
        self.n_latent_var = n_latent_var
        # actor
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

        # critic
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

    def act(self, state, memory, hidden):
        state = torch.from_numpy(state).float().to(device)
        hidden = hidden.to(device)
        action_probs = self.action_head(state)
        action_probs = action_probs.unsqueeze(0).unsqueeze(0)
        action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs[0][0])

        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs[0][0],7,replacement=False)

        # Append each action along with its log_prob and the current state separately
        # Makes the loss function more manageable
        for i in range(7):
            memory.logprobs.append(dist.log_prob(action_indices[i]))
            memory.states.append(state)
            memory.actions.append(action_indices[i])

        return action_indices, hidden

    def evaluate(self, state, action, hidden):
        action_probs = self.action_head(state)
        action_probs = action_probs.unsqueeze(0)
        action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        # Use same distribution as act
        dist = Categorical(action_probs[0])

        # Calculate the expected log_probs for the previous actions
        action_logprobs = dist.log_prob(action)

        # Calculate the entropy from the distribution
        dist_entropy = dist.entropy()

        # Get expected network output
        state_value = self.value_head(state)
        state_value = state_value.unsqueeze(0)
        state_value,_ = self.value_gru(state_value,hidden)
        state_value = self.value_layer(state_value)

        return action_logprobs, torch.squeeze(state_value[0]), dist_entropy