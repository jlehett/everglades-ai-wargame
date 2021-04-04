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

class base_rushV1:
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
        
        
        self.group_location=[1,1,1,1,1,1,1,1,1,1,1,1]
    # end __init__

    def get_action(self, obs):
        #print('!!!!!!! Observation !!!!!!!!')
        ##print(obs)
        #print(obs[0])
        #for i in range(45,101,5):
        #    print(obs[i:i+5])
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        action = np.zeros(self.shape)
        self.update_locations(obs)
        # The next line should really be 0, but there is an env bug that the agent doesn't get
        # to see the 0th observation to make it's first move; one turn gets blown
        if not self.first_turn:
            action = self.act_all_cycle(action)
        else:
            self.first_turn = False
        #print(action)
        
        return action
    # end get_action

    def act_all_cycle(self,actions=np.zeros((7,2))):
        
        self.get_location(2)    
        for i in range(0,7):
            if self.get_location(i) != 11.0:
                actions[i] = [self.group_num, self.node_num]
            
                #self.group_num = ((self.group_num-1) + 1) % self.grouplen + 1
                self.group_num = (self.group_num + 1) % self.grouplen 
                nodetest = ((self.node_num-1) + 1) % self.nodelen + 1
                self.node_num = nodetest if self.group_num == 0 else self.node_num
        #print('---------')
        #print(actions)
        #print('---------')
        return actions
    def update_locations(self,obs):
        i = 45
        index = 0
        while i < 105:
            #self.army[index].pos = obs[i]
            self.group_location[index]=obs[i]
            #print('::::::::::  ',index,'>>>>>>>>>>>>',obs[i])
            #self.army[index].type = obs[i+1]
            #self.army[index].tran = obs[i+3]
            #print(self.army[index])
            i = i + 5
            index = index + 1
    def get_location(self,group_number):
        #print(self.group_location)
        return self.group_location[group_number]
# end class
