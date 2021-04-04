# Standard library imports
import os
import time
import pdb

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

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

class bull_rush:
    def __init__(self, action_space, player_num):
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

        self.node_strat = [2,5,8,11]
        self.strat_index = 0
    # end __init__

    def get_action(self, obs):
        #print('!!!!!!! Observation !!!!!!!!')
        ##print(obs)
        #print(obs[0])
        #for i in range(45,101,5):
        #    print(obs[i:i+5])
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        action = np.zeros(self.shape)

        # The next line should really be 0, but there is an env bug that the agent doesn't get
        # to see the 0th observation to make it's first move; one turn gets blown
        if not self.first_turn:
            if self.strat_index == 8:
                self.strat_index = 0
            action = self.act_bull_rush(action)
        else:
            self.first_turn = False
        #print(action)

        #action[:,0] = [0,1,2,3,4,5,6]
        #action[:,1] = [2,2,2,2,2,2,2]
        return action
    # end get_action

    def act_bull_rush(self,actions=np.zeros((7,2))):
        for i in range(0,7):
            index = self.strat_index // 2
            actions[i] = [self.group_num, self.node_strat[index]]
            #self.group_num = ((self.group_num-1) + 1) % self.grouplen + 1
            self.group_num = (self.group_num + 1) % self.grouplen 
        self.strat_index += 1
        return actions

# end class
