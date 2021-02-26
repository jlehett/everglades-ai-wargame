import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######################
#   PPO Agent Class   #
#######################

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_latent_var,
        lr,
        betas,
        gamma,
        K_epochs, 
        eps_clip
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.memory = Memory()

        # Setup policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        # Do not use RMSProp (learned that the hard way). Adam is the optimizer to use for PPO
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.shape = (7, 2)
        self.loss = 0

        # Set the loss function
        # Only use MSELoss for PPO
        self.MSE = nn.MSELoss()

    def get_action(self, observation):
        action = np.zeros(self.shape)
        chosen_indices = self.policy_old.act(observation,self.memory)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes
        return action

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0

        # Calculate reward discounts for PPO
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding surrogate loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # Original from github branch did not use torch.mean() which left the loss as a tensor when it finished.
            # Found another implementation that uses torch.mean() which has fixed the loss and improved the network
            actor_loss = torch.mean(torch.min(surr1, surr2))
            critic_loss = 0.5*self.MSE(state_values, rewards)
            entropy = 0.01*torch.mean(dist_entropy)
            loss = -actor_loss + critic_loss - entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.loss = loss

        # Copy new weights into old policy:
        hold = self.policy.state_dict
        self.policy_old.load_state_dict(self.policy.state_dict())

####################
#   Memory Class   #
####################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

##########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
            #nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var,n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )
    
    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)

        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs,7,replacement=False)

        # Append each action along with its log_prob and the current state separately
        # Makes the loss function more manageable
        for i in range(7):
            memory.logprobs.append(dist.log_prob(action_indices[i]))
            memory.states.append(state)
            memory.actions.append(action_indices[i])

        return action_indices

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)

        # Use same distribution as act
        dist = Categorical(action_probs)

        # Calculate the expected log_probs for the previous actions
        action_logprobs = dist.log_prob(action)

        # Calculate the entropy from the distribution
        dist_entropy = dist.entropy()

        # Get expected network output
        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy