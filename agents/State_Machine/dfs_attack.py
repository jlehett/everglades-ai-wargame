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
TURNS_FOR_DELAY = 5

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

class dfs_attack:
    def __init__(self, action_space, player_num, map):
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

        # Attack groups
        self.attack_group = [0,1,2,3,4,5,6]
        self.attack_group_2 = [7,8,9,10,11]

        # Indexers
        self.node_index = 1
        self.prev_node = 1
        self.group_index = 0
        self.delay_turns = TURNS_FOR_DELAY
        self.delay = False

        # DFS variables
        self.visited = [False for i in range(11)]
        self.stack = []
        self.stack.append(self.node_index)

    def reset(self):
        self.group_index = 0
        self.delay_turns = TURNS_FOR_DELAY
        self.delay = False
        self.stack = []
        self.stack.append(self.node_index)
        self.visited = [False for i in range(11)]

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
            if self.group_index == 2 or self.delay == True: # Delay actions to allow units to move to correct node
                self.group_index = 0
                if self.delay_turns == 0:
                    self.delay_turns = 10
                    self.delay = False
                else:
                    self.delay = True
                    self.delay_turns -= 1
            else: # Act as long as the units are not already at the enemy base
                action = self.act_dfs_attack()
        else:
            self.first_turn = False
        #print(action)
        
        return action
    # end get_action

    def act_dfs_attack(self,actions=np.zeros((7,2))):
        # If on the second group
        if(self.group_index == 1):
            # Run the attack for group 2
            for i in range(0,len(self.attack_group_2)):
                actions[i] = [self.attack_group_2[i],self.prev_node]
            self.group_index += 1 # increment to next group
        elif(self.group_index == 0 and not self.allvisited()):
            s = self.stack[-1]
            self.prev_node = s
            # Run the attack for group 1
            for i in range(0,len(self.attack_group)):
                actions[i] = [self.attack_group[i],s]
            self.group_index += 1 # increment to next group set

            self.stack.pop()
            if(not self.visited[s-1]):
                self.visited[s-1] = True
            
            for node in NODE_CONNECTIONS[s]:
                if(not self.visited[node - 1]):
                    self.stack.append(node)
        else: # If fully traversed, reset
            self.reset()
        return actions
    
    def allvisited(self):
        flag = True
        for i in self.visited:
            if not i:
                flag = False
        return flag

# end class
    