import torch
import numpy as np
import pickle
import os
from agents.PPO.ActorCritic import ActorCritic
from agents.PPO.PPOMemory import Memory

global device 
device = torch.device('cpu')

#######################
#   PPO Agent Class   #
#######################

class PPOAgent:
    """
    Implementation of PPO with Gated Recurrent Unit
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        n_latent_var = 128,
        lr = 0.001,
        betas = (0.9,0.999),
        gamma = 0.99,
        update_timestep = 2000,
        K_epochs = 4, 
        eps_clip = 0.2,
        lambd = 0.95,
        use_recurrent = False,
        DEVICE = "CPU",
        train = True,
        save_after_episode = 100,
        network_save_name = None
    ):
        """
        Initializes Recurrent PPO

        @param state_dim The size of the state dimesion for everglades
        @param action_dim The size of the action dimension for everglades
        @param n_latent_var The number of nodes in each hidden layer of the Actor Critic Network
        @param lr The set learning rate
        @param betas The beta values for the Adam optimizer
        @param gamma The set gamma values for Generalized Advantage Estimation
        @param update_timestep The number of timesteps until the agent updates
        @param K_epochs The number of epochs to run the loss calculations 
        @param eps_clip The clipping range for PPO ratio calculation
        @param lambd Additional discount factor for Generalized Advantage Estimation
        @param use_recurrent Sets whether or not to use a recurrent architecture
        @param DEVICE The device to use during training/evaluation
        @param train Bool value to determine whether or not to train
        @param save_after_episode The episode number to save at. Ex curr_episode % save_after_episode == 0
        @param network_save_name The save file for the agent
        """
        # Enables GPU Training
        global device
        if DEVICE == 'GPU':
            device = torch.device('cuda')

        # Set recurrent agent or non-recurrent agent
        self.use_recurrent = use_recurrent

        # Set general agent parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.n_latent_var = n_latent_var
        self.shape = (7, 2)
        self.k_wr = 0

        # Set agent saving parameters
        self.network_save_name = network_save_name
        self.save_after_episode = save_after_episode

        # Set agent training parameters
        self.update_timestep = update_timestep
        self.train = train

        # Setup agent memory
        self.memory = Memory()

        # Set up agent policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, device, self.use_recurrent).to(device)
        # Do not use RMSProp (learned that the hard way). Adam is the optimizer to use for PPO
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # Set up agent old_policy
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, device, self.use_recurrent).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Set up the hidden states
        self.hidden = torch.zeros(self.n_latent_var).unsqueeze(0).unsqueeze(0)
        
        # Set up the log variables to be grabbed by the training file
        self.loss = 0
        self.actor_loss = 0
        self.critic_loss = 0
        self.dist_entropy = 0

        # Set the loss function
        # Only use MSELoss for PPO
        self.MSE = torch.nn.MSELoss()

    def get_action(self, observation):
        """
        Gets an agent action at a particular time step

        @param observation: The observation of the agent in the current turn
        @param hidden: The previous hidden state of the agent's GRU
        """
        action = np.zeros(self.shape)
        chosen_indices, self.hidden = self.policy_old.act(observation,self.memory, self.hidden)

        # Unwravel action indices to output to the env
        chosen_units = chosen_indices // 12
        chosen_nodes = chosen_indices % 11

        action[:,0] = chosen_units.cpu()
        action[:,1] = chosen_nodes.cpu()
        return action

    def remember_game_state(self, next_observation = None, reward = None, mask = None):
        """
        Stores game state information in the agent's memory

        @param next_observation The next game observation
        @param reward The reward the agent received
        @param mask The mask for the given state (0 if game is done 1 otherwise)
        """

        # Flip the mask
        mask = 1 if mask == 0 else 0

        # Save once per 7 actions
        for i in range(7):
            self.memory.next_states.append(torch.from_numpy(next_observation).float())
            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(torch.from_numpy(np.asarray(mask)))

    def optimize_model(self):
        """
        Optimizes the model using Generalized Advantage Estimation
        """

        # Pull data from memory
        old_states = torch.stack(self.memory.states).to(device).detach()
        next_states = torch.stack(self.memory.next_states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        mask = torch.stack(self.memory.is_terminals).to(device).detach()
        reward = torch.from_numpy(np.asarray(self.memory.rewards)).to(device).detach()
        
        # Set the hidden states
        hidden = torch.zeros(old_states.size(0),self.n_latent_var).unsqueeze(0).to(device)

        # Calculate values for Generalized Advantage Estimation
        _,values,_ = self.policy.evaluate(old_states, old_actions, hidden)
        values = values.detach()

        # Calculate the Advantage
        rewards = self.calc_gae(reward, values, mask)

        # Calculate the losses
        self.calc_losses(old_states, old_actions, old_logprobs, rewards, hidden)

        # Reset the hidden states
        self.hidden = torch.zeros(self.n_latent_var).unsqueeze(0).unsqueeze(0)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def calc_gae(self, reward, values, mask):
        """
        Calculates the Generalized Advantage Estimation given current memory

        @param reward The batch reward tensor from the agent's memory
        @param values The calculated values from the critic network
        @param mask The batch of mask state values (1 if not done, 0 otherwise)
        """

        rewards = []
        gae = 0
        last_value = values[len(values) - 1]

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

        return rewards

    def calc_losses(self, old_states, old_actions, old_logprobs, rewards, hidden):
        """
        Calculates the Actor, Critic and Total losses for RPPO and takes an optimization step for K_Epochs

        @param old_states The batch of previous states stored in the agent's memory
        @param old_actions The batch of old actoins taken by the agent stored in the agent's memory
        @param old_logprobs The batch of old logprobs stored in the agent's memory
        @param rewards The rewards calculated by Generalized Advantage Estimation
        @param hidden The hidden states to use in loss calculation
        """

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

            # Calculate the losses
            actor_loss = torch.min(surr1, surr2)
            critic_loss = self.MSE(state_values, rewards) * 0.5
            entropy = 0.01*dist_entropy
            loss = -actor_loss + critic_loss - entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            # Set the class loss values to be grabbed by training file for display
            self.loss = loss.detach().cpu()
            self.actor_loss = actor_loss.mean().detach().cpu()
            self.critic_loss = critic_loss.detach().cpu()
            self.dist_entropy = entropy.mean().detach().cpu()
        pass

    def end_of_episode(self, episode, k_wr):
        """
        Handles end of episode logic for the agent

        @param episode The current episode the agent is on
        @param k_wr The win rate after k episodes (used for LR decay)
        """
        # Handle end of episode logic
        if self.train:
            self.end_of_episode_train(episode, k_wr)
        

    def end_of_episode_train(self, episode, k_wr):
        """
        Handles end of episode logic while agent is training

        @param episode The current episode the agent is on
        @param k_wr The win rate after k episodes (used for LR decay)
        """
        # Handle end of episode while training
        if episode % self.save_after_episode == 0:
            self.save_network(episode)
        
        # Decay the learning rate if new k_wr is less than historic k_wr
        if episode % self.save_after_episode == 0 and k_wr <= self.k_wr:
            self.lr = self.lr / 2
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
    
    def save_network(self, episodes):
        """
        Saves the network's state dict, epsilon value, and episode count to the specified file.

        @param episodes The number of episodes that have elapsed since the current training session began
        """
        if self.network_save_name:
            save_file = open(os.getcwd() + self.network_save_name + '.pickle', 'wb')
            pickle.dump({
                'type': 'PPO',
                'policy_old_state_dict': self.policy_old.state_dict(),
                'policy_state_dict': self.policy.state_dict(),
                'update_timestep': self.update_timestep,
                'use_recurrent': self.use_recurrent,
                'n_latent_var': self.n_latent_var,
                'lr': self.lr,
                'betas': self.betas,
                'k_epochs': self.K_epochs,
                'eps_clip': self.eps_clip,
                'gamma': self.gamma,
                'lambda': self.lambd
            }, save_file)
            save_file.close()
            print('Saved Network')
        else:
            print('Save Failed - Save File Not Specified')