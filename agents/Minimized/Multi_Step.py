from collections import namedtuple, deque
import random
import numpy as np

## DM PER EDIT ##
#from agents.Minimized.DQNAgent import DQNAgent

OBSERVATION_SPACE = (12, 59)

"""
    The NStepModule will contain necessary util functions for implementing N-Step Learning
    on a DQN agent.
"""
class NStepModule:
    def __init__(self, n, discount_rate, memory_size, policy_net):
        # Stores the modified replay memory constructed by the NStepModule to be used in the DQN agent
        self.replay_memory = NStepReplayMemory(memory_size, policy_net)
        # Tracks the game state across the entire length of a single game
        self.game_memory = [] # Items consist of tuples: (state, action, reward)
        # The discount rate to use in computing future rewards (should probably be the same as the discount rate of the DQN)
        self.discount_rate = discount_rate
        # The number of steps to use in n-step learning
        self.n = n
    
    def trackGameState(self, swarm_states, action, reward):
        """
            @Public
            Add the swarm game states for the current time step to the game memory

            @param swarm_states Array of each swarm's observations indexed by swarm number
            @param action 7x2 action array composed of the 7 actions taken, each represented by the tuple (swarm_number 0-indexed, node_number 1-indexed)
            @param reward Float value containing the reward received by the agent for taking specified actions at the current step
        """
        self.game_memory.append((swarm_states, action, reward))

    def addGameToReplayMemory(self, policy_net):
        """
            @Public
            Once a game has completed, we need to add all of the experience of the game to the
            NStepReplayMemory. Automatically resets the game memory.
        """
        doesNotHitDone = True
        for step_num, game_step in enumerate(self.game_memory):
            previous_state_swarms = game_step[0]
            actions = self.game_memory[step_num][1]
            actualSummedReward = self.getSummedReward(step_num)
            next_state_swarms = np.zeros(OBSERVATION_SPACE)
            if step_num + self.n < len(self.game_memory):
                next_state_swarms = self.game_memory[step_num + self.n][0]
            else:
                doesNotHitDone = False
            # Add per-swarm observations if action was taken by swarm at this state
            for swarm_num, swarm_state in enumerate(previous_state_swarms):
                # Determine the node the swarm moved to this step
                node_moved_to = -1
                for action in actions:
                    if action[0] == swarm_num:
                        node_moved_to = action[1] - 1
                        break
                # If the node_moved_to is -1, the swarm did not take an action;
                # If the swarm DID take an action, add it to replay memory
                if node_moved_to != -1:
                    self.replay_memory.push( policy_net, swarm_state, int(node_moved_to), next_state_swarms, actualSummedReward, doesNotHitDone, qnext)
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
    ## DM PER EDIT ##
    def __init__(self, capacity, policy_net):
        self.capacity = capacity
        self.memory = []
        self.priority = deque(maxlen=10000)
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('swarm_obs', 'swarm_action', 'next_state_swarms', 'reward', 'doesNotHitDone'))
        
        

    ## DM PER EDIT ##
    def push(self, policy_net, *args):
        print(policy_net)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # self.replay_memory.push(previous_state_swarms, actions, next_state_swarms, actualSummedReward, hitsDone)
        self.prioritize(policy_net, *args)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    ## DM PER EDIT ##
    def prioritize(self, policy_net, state, action, next_state, reward, done, qnext):
        print(policy_net(next_state)[0])
        q_next = reward + 0.9 * np.max(policy_net(next_state)[0])
        q = policy_net(state)[0][action]
        p = (np.abs(q_next-q)+ (np.e ** -10)) ** alpha
        self.priority.append(p)

    def sample(self, batch_size):
        p_sum = np.sum(self.replay_memory.priority)
        prob = self.replay_memory.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k=self.batch_size, weights=prob)
        importance = (1/prob) * (1/len(self.replay_memory.priority))
        importance = np.array(importance)[sample_indices]
        samples = np.array(self.replay_memory)[sample_indices]
        return samples, importance

    def __len__(self):
        return len(self.memory)