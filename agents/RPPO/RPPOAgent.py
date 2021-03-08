import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from agents.RPPO.IntrinsicCuriosity import ICM

device = torch.device('cpu')

#######################
#   PPO Agent Class   #
#######################

TEMP_START = 1
TEMP_END = 1.0
TEMP_DECAY = 0.00005

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
        intr_reward_strength,
        icm_batch_size,
        target_kl,
        lambd,
        use_icm
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.intr_reward_strength = intr_reward_strength
        self.n_latent_var = n_latent_var

        self.memory = Memory()

        # Setup policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        # Do not use RMSProp (learned that the hard way). Adam is the optimizer to use for PPO
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        #self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        #self.policy_old.load_state_dict(self.policy.state_dict())
        self.shape = (7, 2)
        self.loss = 0

        self.icm = ICM(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=lr, betas=betas)
        self.inv_loss = 0
        self.forward_loss = 0
        self.icm_batch_size = icm_batch_size
        self.use_icm = use_icm

        self.temperature = 0
        self.steps_done = 0

        self.target_kl = target_kl

        # Set the loss function
        # Only use MSELoss for PPO
        self.MSE = nn.MSELoss()

    def get_action(self, observation, hidden):

        self.temperature = TEMP_END + (TEMP_START - TEMP_END) * np.exp(self.steps_done * -TEMP_DECAY)
        self.steps_done += 1

        action = np.zeros(self.shape)
        chosen_indices, hidden = self.policy.act(observation,self.memory, self.temperature, hidden)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units
        action[:,1] = chosen_nodes
        return action, hidden

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        gae = 0

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        next_states = torch.stack(self.memory.next_states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        mask = torch.stack(self.memory.is_terminals).to(device).detach()
        reward = torch.from_numpy(np.asarray(self.memory.rewards)).to(device).detach()

        # Get the intrinsic reward
        with torch.no_grad():
            intr_reward, self.inv_loss,self.forward_loss = self.icm(old_actions, old_states, next_states, mask)
        intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)

        if not self.use_icm:
            intr_rewards *= 0

        hidden = torch.zeros(old_states.size(0),self.n_latent_var).unsqueeze(0)
        _,values,_ = self.policy.evaluate(old_states, old_actions, self.temperature, hidden)
        # Calculate the GAE
        for i in reversed(range(len(self.memory.rewards))):
            mask_i = 0 if mask[i] > 0 else 1
            delta = (reward[i] + intr_rewards[i]) + (self.gamma * values[i-1] * mask_i) - values[i]
            gae = delta + (self.gamma * self.lambd * mask_i * gae)
            #discounted_reward = (reward + intr_reward) + (self.gamma * discounted_reward)
            rewards.insert(0, gae + values[i])

        rewards = torch.stack(rewards)
        rewards = rewards - values
        # Normalizing the rewards:
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.detach()

        batch_idx = np.arange(len(old_states))
        stop_early = False
        # Optimize policy for K epochs:
        for k in range(self.K_epochs):

            # For kl divergence early stopping
            logprobs, _, _ = self.policy.evaluate(old_states, old_actions, self.temperature, hidden)
            approx_kl = (old_logprobs - logprobs).mean().item()
            if approx_kl > 1.5 * self.target_kl and not stop_early:
                print('Stopping on epoch {} due to reach max kl'.format(k))
                stop_early = True

            np.random.shuffle(batch_idx)
            # Run on minibatches
            for j in range(len(self.memory.rewards) // self.icm_batch_size):
                sample_idx = batch_idx[self.icm_batch_size * j:self.icm_batch_size * (j + 1)]
                # Evaluating old actions and values:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[sample_idx], old_actions[sample_idx], self.temperature, hidden[:,sample_idx,:])

                # Finding the ratio (pi_theta / pi_theta_old):
                ratios = torch.exp(logprobs - old_logprobs[sample_idx].detach())

                # Finding surrogate loss:
                advantages = rewards #- state_values.detach()
                surr1 = ratios * advantages[sample_idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[sample_idx]
                # Original from github branch did not use torch.mean() which left the loss as a tensor when it finished.
                # Found another implementation that uses torch.mean() which has fixed the loss and improved the network
                actor_loss = torch.mean(torch.min(surr1, surr2))
                critic_loss = 0.5*self.MSE(state_values, rewards[sample_idx])
                entropy = 0.01*torch.mean(dist_entropy)
                loss = -actor_loss + critic_loss - entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                self.loss = loss

                k_epochs = k
            if stop_early:
                break
        
        if self.use_icm:
            self._icm_update(old_states, next_states, old_actions, mask, k_epochs)

        # Copy new weights into old policy:
        #self.policy_old.load_state_dict(self.policy.state_dict())

    def _icm_update(self, curr_states, next_states, actions, mask, k_epochs):
        if self.icm_batch_size > curr_states.size(0):
            pass

        epoch_forw_loss = 0
        epoch_inv_loss = 0
        for _ in range(k_epochs):
            indexes = np.random.permutation(actions.size(0))
            for i in range(0, len(indexes), self.icm_batch_size):
                batch_ind = indexes[i:i + self.icm_batch_size]
                batch_curr_states = curr_states[batch_ind, :]
                batch_next_states = next_states[batch_ind, :]
                batch_actions = actions[batch_ind]
                batch_mask = mask[batch_ind]

                _, inv_loss, forw_loss = self.icm(batch_actions,
                                                  batch_curr_states,
                                                  batch_next_states,
                                                  batch_mask)
                epoch_forw_loss += forw_loss.item()
                epoch_inv_loss += inv_loss.item()
                unclip_intr_loss = 10 * (0.2 * forw_loss + 0.8 * inv_loss)

                # take gradient step
                self.optimizer_icm.zero_grad()
                unclip_intr_loss.backward()
                self.optimizer_icm.step()

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

    def act(self, state, memory, temperature, hidden):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_head(state)
        action_probs = action_probs.unsqueeze(0).unsqueeze(0)
        action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs[0][0] / temperature)

        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        action_indices = torch.multinomial(action_probs[0][0] / temperature,7,replacement=False)

        # Append each action along with its log_prob and the current state separately
        # Makes the loss function more manageable
        for i in range(7):
            memory.logprobs.append(dist.log_prob(action_indices[i]))
            memory.states.append(state)
            memory.actions.append(action_indices[i])

        return action_indices, hidden

    def evaluate(self, state, action, temperature, hidden):
        action_probs = self.action_head(state)
        action_probs = action_probs.unsqueeze(0)
        action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        # Use same distribution as act
        dist = Categorical(action_probs[0] / temperature)

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