####################
#   Memory Class   #
####################

class Memory:
    """
    Memory class for PPO
    """
    def __init__(self):
        """
        Initilizes the memory
        """
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hiddens = []

    def clear_memory(self):
        """
        Completely wipes the memory for reuse
        """
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hiddens[:]