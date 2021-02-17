# Standard library imports
import os
import time
import pdb
import random

# Specialized imports
import numpy as np
#import gym
#import gym_everglades

NODE_CONNECTIONS = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 4, 5, 6, 7],
    4: [1, 3, 7],
    5: [2, 3, 8, 9],
    6: [3, 9],
    7: [3, 4, 9, 10],
    8: [5, 9, 11],
    9: [5, 6, 7, 8, 10],
    10: [7, 9, 11],
    11: [8, 10]
}


NUM_GROUPS = 12
TURNS_FOR_DELAY = 10

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

class bull_rush:
    def __init__(self, action_space, player_num,map_name):
        self.action_space = action_space
        self.num_groups = NUM_GROUPS

        self.num_actions = action_space
        self.shape = (self.num_actions, 2)

        self.first_turn = True
        self.steps = 0
        self.player_num = player_num
        #print('player_num: {}'.format(player_num))

        # Types:
        #   0 - Controller
        #   1 - Striker
        #   2 - Tank
        
        self.grouplen = NUM_GROUPS
        self.nodelen = len(NODE_CONNECTIONS)
        self.group_num = 1
        self.node_num = 2

        # Attack strategies
        self.node_strat = [2,5,8,11]
        self.node_strat_2 = [4,7,10,11]
        self.set_strat = 1

        # Attack groups
        self.attack_group = [0,1,2,3,4,5,6]
        self.attack_group_2 = [7,8,9,10,11]

        # Indexers
        self.group_index = 0
        self.strat_index = 0
        self.delay_turns = TURNS_FOR_DELAY
        self.delay = False

    # end __init__

    def reset(self):
        #Do reset stuffs
        self.strat_index = 0
        self.group_index = 0
        self.delay_turns = TURNS_FOR_DELAY
        self.delay = False

        # Set the attack strategy
        sample = random.random()
        if sample < 0.5:
            self.set_strat = 1
        else:
            self.set_strat = 2

    def get_action(self, obs):
        action = np.zeros(self.shape)

        # The next line should really be 0, but there is an env bug that the agent doesn't get
        # to see the 0th observation to make it's first move; one turn gets blown
        if not self.first_turn:
            if self.group_index == 2 or self.delay == True: # Delay actions to allow units to move to correct node
                self.group_index = 0
                if self.delay_turns == 0:
                    self.delay_turns = 10
                    self.delay = False
                else:
                    self.delay = True
                    self.delay_turns -= 1
            elif self.strat_index < 8: # Act as long as the units are not already at the enemy base
                if self.set_strat == 1:
                    action = self.act_bull_rush_strat1(action)
                elif self.set_strat == 2:
                    action = self.act_bull_rush_strat2(action)
        else:
            self.first_turn = False

        return action
    # end get_action

    # Use attack strategy 1
    def act_bull_rush_strat1(self,actions=np.zeros((7,2))):
        index = self.strat_index // 2
        if(self.group_index == 0):
            for i in range(0,len(self.attack_group)):
                actions[i] = [self.attack_group[i],self.node_strat[index]]
            self.group_index += 1
        elif(self.group_index == 1):
            for i in range(0,len(self.attack_group_2)):
                actions[i] = [self.attack_group_2[i],self.node_strat[index]]
            self.group_index += 1
        self.strat_index += 1
        return actions

    # Use attack strategy 2
    def act_bull_rush_strat2(self,actions=np.zeros((7,2))):
        index = self.strat_index // 2 # Set the node to attack
        if(self.group_index == 0): # First group attacks
            for i in range(0,len(self.attack_group)):
                actions[i] = [self.attack_group[i],self.node_strat_2[index]]
            self.group_index += 1
        elif(self.group_index == 1): # Second group attacks
            for i in range(0,len(self.attack_group_2)):
                actions[i] = [self.attack_group_2[i],self.node_strat_2[index]]
            self.group_index += 1
        self.strat_index += 1 # Increment the node to attack
        return actions

# end class
