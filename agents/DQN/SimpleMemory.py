import random
from collections import namedtuple
### DEFINE REPLAY MEMORY ###

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Simple Replay Memory for DQN
    """
    # Simple Replay Memory
    def __init__(self, capacity):
        """
        Initializes the replay memory with a set capacity

        @param capacity The capacity of the replay memory
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition in the memory

        @param *args A transition to be stored in the memory
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Get a random sample of transitions from the memory of size batch_size

        @param batch_size The size of the batch to be pulled from memory
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the length of the memory
        """
        return len(self.memory)