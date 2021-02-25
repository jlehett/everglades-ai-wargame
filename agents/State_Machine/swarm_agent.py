

# Imports
import os
import numpy as np
import time

#import gym
#import gym_everglades

#from gym_everglades.envs.everglades_env import UnitState, GroupState

# Global Definitions
# Map connections
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

ATTACK_LIST = [1,2,4,5,7,8,10,11] # only use sentries and controllers

NUM_GROUPS = 12

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

# Unit Class
class Unit:
    def __init__(self):
        self.pos = 1        # Last known node*
        self.type = 0       # 0: Controller, 1: Striker, 2: Tank 
        self.tran = 0       # In transit

    def __str__(self):
        s = 'Pos: ' + str(self.pos) + ' type: ' + str(self.type) + ' tran: ' + str(self.tran)
        return s



# Agent Class
class SwarmAgent:
    def __init__(self, action_space, player_num):
        # Basic Agen Information ########################
        self.action_space = action_space                #
        self.num_groups = NUM_GROUPS                    #
        self.num_nodes = len(NODE_CONNECTIONS)          #
        self.num_actions = action_space   #
        self.shape = (self.num_actions, 2)              #
        #################################################

        self.army = [Unit() for _ in range(12)]

    def update_army(self, obs):
        #print('----Updating Army----')
        i = 45
        index = 0
        while i < 105:
            self.army[index].pos = obs[i]
            self.army[index].type = obs[i+1]
            self.army[index].tran = obs[i+3]
            #print(self.army[index])
            i = i + 5
            index = index + 1


    def get_action(self, obs):
        #print('----Making Action List----')
        a = np.array([0,1])
        action = np.tile(a, (7, 1))

        self.update_army(obs)

        temp_list = ATTACK_LIST
        np.random.shuffle(temp_list)

        i = 0
        for x in temp_list:
            # s = 'i=' + str(i) + ' x=' + str(x)
            # print(s)
            if i == 7:
                break
            # print(self.army[x].tran)
            if self.army[x].tran == 0:
                action[i] = [x, max(NODE_CONNECTIONS.get(self.army[x].pos))]
                i = i + 1

        #print('----Printing Actions----')
        #print(action)
        return action
