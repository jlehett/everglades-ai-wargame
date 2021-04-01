import pickle
import numpy as np

class AgentStatistics:
    """
    This class contains all necessary statistical information for a given agent and saves it to a file
    """
    def __init__(self, agent_name, n_episodes, k, save_file = None):
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
        self.epsilonVals = []
        self.lossVals = []

        # PPO Loss stats
        self.actorLossVals = []
        self.criticLossVals = []

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
                'epsilon': self.epsilonVals,
                'loss': self.lossVals,
                'actor_loss': self.actorLossVals,
                'critic_loss': self.criticLossVals
            }, save_file)
            save_file.close()
            print('Saved Statistics')
        else:
            print('Save Failed - Save File Not Specified')