import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from IntrinsicCuriosity import ICM

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
        use_icm,
        use_gru
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.intr_reward_strength = intr_reward_strength
        self.n_latent_var = n_latent_var
        self.use_gru = use_gru

        self.memory = Memory()

        # Setup policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, self.use_gru).to(device)
        # Do not use RMSProp (learned that the hard way). Adam is the optimizer to use for PPO
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.shape = (7, 2)
        self.loss = 0
        self.actor_loss = 0
        self.critic_loss = 0

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

        action = chosen_indices.item()
        return action, hidden

    def optimize_model(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        gae = 0

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        next_states = torch.stack(self.memory.next_states).to(device).detach()
        old_actions = torch.from_numpy(np.asarray(self.memory.actions)).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        mask = torch.stack(self.memory.is_terminals).to(device).detach()
        reward = torch.from_numpy(np.asarray(self.memory.rewards)).to(device).detach()
        hidden_states = torch.cat(self.memory.hidden).to(device).detach()

        # Get the intrinsic reward
        if self.use_icm:
            with torch.no_grad():
                intr_reward, self.inv_loss,self.forward_loss = self.icm(old_actions, old_states, next_states, mask)
            intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)

        #if not self.use_icm:
            #intr_rewards *= 0

        hidden_states = torch.zeros(old_states.size(0),self.n_latent_var).unsqueeze(0)
        _,values,_ = self.policy.evaluate(old_states, old_actions, self.temperature, hidden_states)
        values = values.detach()
        last_value = values[len(values) - 1]
        # Calculate the Advantage
        for i in reversed(range(len(self.memory.rewards))):
            ### Markovian Discounted Rewards ###
            #if mask[i] < 1:
            #    discounted_reward = 0
            #discounted_reward = (reward[i]) + (self.gamma * discounted_reward)
            #rewards.insert(0,discounted_reward)
            ### End Markovian Discounted Rewards ###

            ### Generalized Advantage Estimation ###
            delta = (reward[i]) + (self.gamma * last_value * mask[i]) - values[i]
            gae = delta + (self.gamma * self.lambd * mask[i] * gae)
            last_value = values[i]
            rewards.insert(0, gae + values[i])
            ### End Generalized Advantage Estimation ###

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device) - values[:]
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        #rewards = rewards.detach()

        stop_early = False
        # Optimize policy for K epochs:
        for k in range(self.K_epochs):

            # For kl divergence early stopping
            #logprobs, _, _ = self.policy.evaluate(old_states, old_actions, self.temperature, hidden_states)
            #approx_kl = (old_logprobs - logprobs).mean().item()
            #if approx_kl > 1.5 * self.target_kl and not stop_early:
            #    print('Stopping on epoch {} due to reach max kl'.format(k))
            #    stop_early = True

            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, self.temperature, hidden_states)

            # Finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding surrogate loss:
            advantages = rewards #- state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # Original from github branch did not use torch.mean() which left the loss as a tensor when it finished.
            # Found another implementation that uses torch.mean() which has fixed the loss and improved the network
            actor_loss = torch.min(surr1, surr2)
            critic_loss = self.MSE(state_values, rewards.float()) * 1e-1#0.5
            entropy = 0.01*dist_entropy
            loss = -actor_loss + critic_loss - entropy
            #loss = loss.float()

            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.loss = loss.detach()
            self.actor_loss = actor_loss.mean().detach()
            self.critic_loss = critic_loss.detach()

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
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]

##########################
#   Actor Critic Class   #
##########################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, use_gru):
        super(ActorCritic, self).__init__()
        self.num_layers = 2
        self.use_gru = use_gru
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
        if self.use_gru:
            action_probs = action_probs.unsqueeze(0).unsqueeze(0)
            action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        if self.use_gru:
            action_probs = action_probs[0][0]

        # Uses Boltzmann style exploration by sampling from distribution
        dist = Categorical(action_probs)
        action_indices = dist.sample()

        # Multinomial uses the same distribution as Categorical but allows for sampling without replacement
        # Enables us to grab non-duplicate actions faster
        #action_indices = torch.multinomial(action_probs / temperature,1,replacement=False)

        # Append each action along with its log_prob and the current state separately
        # Makes the loss function more manageable
        memory.logprobs.append(dist.log_prob(action_indices))
        memory.states.append(state)
        memory.actions.append(action_indices.item())
        memory.hidden.append(hidden)

        return action_indices, hidden

    def evaluate(self, state, action, temperature, hidden):
        # Fix the hidden tensor to be batch second
        #hidden = hidden.reshape(1, hidden.size(0), hidden.size(2))
        action_probs = self.action_head(state)
        if self.use_gru:
            action_probs = action_probs.unsqueeze(0)
            action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)

        if self.use_gru:
            action_probs = action_probs[0]

        # Use same distribution as act
        dist = Categorical(action_probs / temperature)

        # Calculate the expected log_probs for the previous actions
        action_logprobs = dist.log_prob(action)

        # Calculate the entropy from the distribution
        dist_entropy = dist.entropy()

        # Get expected network output
        state_value = self.value_head(state)
        if self.use_gru:
            state_value = state_value.unsqueeze(0)
            state_value,_ = self.value_gru(state_value,hidden)
        state_value = self.value_layer(state_value)

        if self.use_gru:
            state_value = state_value[0]

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def get_fresh_hidden(self, state, hidden):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_head(state)
        if self.use_gru:
            action_probs = action_probs.unsqueeze(0).unsqueeze(0)
            action_probs,hidden = self.action_gru(action_probs, hidden)
        action_probs = self.action_layer(action_probs)
        return hidden