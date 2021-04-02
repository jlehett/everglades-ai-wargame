import pickle
import numpy as np
import os

class AgentStatistics:
    """
    This class contains all necessary statistical information for a given agent and saves it to a file
    """
    def __init__(self, agent_name = None, n_episodes = None, k = None, save_file = None):
        """
        Sets up the Agent Statistics class for saving statistical information

        @param agent_name The name of the agent whose stats these belong
        @param n_episodes The number of episodes the agent was trained/tested
        @param k The set number of episodes to show win rates for
        """

        # Basic class members
        self.agent_name = agent_name
        self.n_episodes = n_episodes
        self.k = k

        # Save file
        self.save_file = save_file

        # Win/Loss/Tie Stats
        self.ties = 0
        self.losses = 0
        self.wins = 0

        # Score stats
        self.scores = []
        self.short_term_scores = [0.5] # Average win rates per k episodes
        
        # Epsilong/Loss stats
        self.epsilons = []
        self.network_loss = []

        # DQN Q Values
        self.q_values = []

        # PPO Loss stats
        self.actor_loss = []
        self.critic_loss = []

    def save_stats(self):
        """
        Saves the agent's statistics.
        """
        if self.save_file:
            save_file = open(self.save_file + '.pickle', 'wb')
            pickle.dump({
                'agent_name': self.agent_name,
                'n_episodes': self.n_episodes,
                'k': self.k,
                'wins': self.wins,
                'losses': self.losses,
                'ties': self.ties,
                'scores': self.scores,
                'short_term_scores': self.short_term_scores,
                'epsilon': self.epsilons,
                'loss': self.network_loss,
                'actor_loss': self.actor_loss,
                'critic_loss': self.critic_loss,
                'q_values': self.q_values
            }, save_file)
            save_file.close()
            print('Saved Statistics')
        else:
            print('Save Failed - Save File Not Specified')

    def load_stats(self, save_file = None):
        """
        Load's data into the statistics model

        @param save_file The save_file to load data from
        """

        # Get the contents of the save file
        save_file_data = self.get_file_contents(save_file)

        # Load the contents into the statistics model
        self.agent_name         = save_file_data['agent_name']
        self.n_episodes         = save_file_data['n_episodes']
        self.k                  = save_file_data['k']
        self.wins               = save_file_data['wins']
        self.losses             = save_file_data['losses']
        self.ties               = save_file_data['ties']
        self.scores             = save_file_data['scores']
        self.short_term_scores  = save_file_data['short_term_scores']
        self.epsilons           = save_file_data['epsilon']
        self.network_loss       = save_file_data['loss']
        self.actor_loss         = save_file_data['actor_loss']
        self.critic_loss        = save_file_data['critic_loss']
        self.q_values           = save_file_data['q_values']

    def get_file_contents(self, save_file_path):
        """
        Helper method to get the contents of the statistics save file

        @param save_file_path The path to the statistics save file
        """
        if not os.path.exists(save_file_path + '.pickle'):
            print('Save file does not exist; exiting...')
            return None

        # Load the saved model
        save_file = open(save_file_path + '.pickle', 'rb')
        save_file_data = pickle.load(save_file)
        save_file.close()

        return save_file_data