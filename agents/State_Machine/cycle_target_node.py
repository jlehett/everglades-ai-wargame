import os
import time
import pdb
import numpy as np

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

TAR_NODE = {
    # key is target node, entry is next node from index node
    # ex to get to 1 from 11 go to 8  1[(11 - 1)] = 8 
    # subtract 1 for shift in array
    # also entry is -1 if targetting current node
    1: [-1, 1, 4, 1, 2, 3, 4, 5, 7, 7, 8],
    2: [2, -1, 2, 1, 2, 3, 3, 5, 5, 7, 8],
    3: [4, 3, -1, 3, 3, 3, 3, 5, 5, 7, 8],
    4: [4, 1, 4, -1, 3, 3, 4, 5, 7, 7, 10],
    5: [2, 5, 5, 3, -1, 9, 3, 5, 5, 9, 8],
    6: [2, 3, 6, 3, 9, -1, 9, 9, 6, 9, 10],
    7: [4, 3, 7, 7, 3, 3, -1, 9, 7, 7, 10],
    8: [2, 5, 5, 7, 8, 9, 9, -1, 8, 11, 8],
    9: [2, 5, 7, 7, 9, 9, 9, 9, -1, 9, 8],
    10: [4, 3, 7, 7, 9, 9, 10, 9, 10, -1, 10],
    11: [2, 5, 7, 7, 8, 9, 10, 11, 10, 11, -1]

    #TODO Dijkstra's 
}

NUM_GROUPS = 12

ENV_MAP = {
    'everglades': 'Everglades-v0',
    'everglades-vision': 'EvergladesVision-v0',
    'everglades-stoch': 'EvergladesStochastic-v0',
    'everglades-vision-stoch': 'EvergladesVisionStochastic-v0',
}

class Cycle_Target_Node:
    def __init__(self, action_space, player_num):
        self.action_space = action_space
        self.num_groups = NUM_GROUPS
        self.num_actions = action_space
        self.shape = (self.num_actions, 2)
        self.first_turn = True
        self.steps = 0
        self.player_num = player_num

        # Types:
        #   0 - Controller
        #   1 - Striker
        #   2 - Tank
        
        self.grouplen = NUM_GROUPS
        self.nodelen = len(NODE_CONNECTIONS)
        self.group_num = 1
        self.node_num = 2

        # determines which node to rush towards
        self.tarNode = 11
        # stores group locations
        self.group_location=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # TODO could make the script more efficient by checking if each group
        # can move instead of just trying to move the next seven groups
        self.group_movable=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_action(self, obs):
        action = np.zeros(self.shape)

        # the next line should really be 0, but there is an env bug that the agent doesn't get
        # to see the 0th observation to make it's first move; one turn gets blown
        if not self.first_turn:
            self.update_groups(obs)
            action = self.node_rush(action, self.isNodeControlled(obs))
        else:
            self.first_turn = False
        
        return action

    def act_all_cycle(self,actions = np.zeros((7, 2))):
        for i in range(0, 7):
            actions[i] = [self.group_num, self.node_num]
            self.group_num = (self.group_num + 1) % self.grouplen 
            nodetest = (self.node_num) % self.nodelen + 1
            self.node_num = nodetest if self.group_num == 0 else self.node_num
        return actions

    # parameter nodeControl is determined by calling isNodeControlled function
    def node_rush(self, actions = np.zeros((7, 2)), nodeControl = False):
        if nodeControl:
            return self.act_all_cycle(actions)
        else:
            # determine next nodes for target
            nextNodes = TAR_NODE[self.tarNode]
            for i in range(0, 7):
                # find node to go to for current group number
                curNode = int(self.group_location[self.group_num])
                nextNode = nextNodes[curNode - 1]

                # from original all cycle code
                actions[i] = [self.group_num, nextNode]
                self.group_num = (self.group_num + 1) % self.grouplen
            return actions
    
    def update_groups(self, obs):
        i = 45
        index = 0
        while i < 101:
            self.group_location[index] = obs[i]
            self.group_movable[index] = obs[i + 3]
            i += 5
            index += 1

    def isNodeControlled(self, obs):
        desiredControlLevel = 75
        if (obs[self.tarNode * 4 - 1] >= desiredControlLevel):
            return True
        else:
            return False
 
