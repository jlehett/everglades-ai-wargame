from collections import namedtuple, deque
import random
import numpy as np

OBSERVATION_SPACE = 105

"""
    The NStepModule will contain necessary util functions for implementing N-Step Learning
    on a DQN agent.
"""
class NStepModule:
    def __init__(self, n, discount_rate, memory_size):
        # Stores the modified replay memory constructed by the NStepModule to be used in the DQN agent
        self.replay_memory = NStepReplayMemory(memory_size)
        # Tracks the game state across the entire length of a single game
        self.game_memory = [] # Items consist of tuples: (state, action, reward)
        # The discount rate to use in computing future rewards (should probably be the same as the discount rate of the DQN)
        self.discount_rate = discount_rate
        # The number of steps to use in n-step learning
        self.n = n
    
    def trackGameState(self, state, action, reward):
        """
            @Public
            Add the game state for the current time step to the game memory
        """
        self.game_memory.append((state, action, reward))

    def addGameToReplayMemory(self):
        """
            @Public
            Once a game has completed, we need to add all of the experience of the game to the
            NStepReplayMemory. Automatically resets the game memory.
        """
        hitsDone = False
        for step_num, game_step in enumerate(self.game_memory):
            previous_state = game_step[0]
            actions = self.game_memory[step_num][1]
            actualSummedReward = self.getSummedReward(step_num)
            next_state = np.zeros(OBSERVATION_SPACE)
            if step_num + self.n < len(self.game_memory):
                next_state = self.game_memory[step_num + self.n][0]
            else:
                hitsDone = True
            self.replay_memory.push(previous_state, actions, next_state, actualSummedReward, hitsDone)
        # Reset the game memory
        self.resetGameMemory()

    def isMemoryLargeEnoughToTrain(self, batch_size):
        """
            @Public
            Returns true if we can start training or false otherwise.
        """
        return len(self.replay_memory) >= batch_size

    def sampleReplayMemory(self, batch_size):
        """
            @Public
            Samples the replay memory for traning data equal to the batch_size.
        """
        return self.replay_memory.sample(batch_size)

    def resetGameMemory(self):
        """
            @Private -- called by addGameToReplayMemory
            Reset the memory for a new game
        """
        self.game_memory = []

    def getSummedReward(self, step_num):
        """
            @Private -- called by addGameToReplayMemory
            Grab the summed and discounted reward to use in the loss function for training the network
        """
        sum_total = self.game_memory[step_num][2]
        for k in range(self.n):
            if step_num + k + 1 < len(self.game_memory):
                sum_total += (self.discount_rate ** k) * self.game_memory[step_num + k + 1][2]
            else:
                sum_total += 0 # Game is over, no more reward to be received.
        return sum_total

    
"""
    The NStepReplayMemory will be used to store the modified N-Step Replay Memory to be used in the
    DQN agent.
"""
class NStepReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'hitsDone'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)