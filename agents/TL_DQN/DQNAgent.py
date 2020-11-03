import os
import numpy as np
import time
import json
import random
import torch
import torch.nn as nn
from collections import deque
import time

OBSERVATION_DIM = 58 #105 or 67
NUM_UNIT_GROUPS = 12
NUM_NODES = 11
NUM_ACTIONS = 7
MAX_NUM_CONNECTIONS = 6

POSSIBLE_ACTIONS = (NUM_UNIT_GROUPS, NUM_NODES)

import torch
import random
import numpy as np

class DQNAgent:
    def __init__(
        self,
        env,
        map_name,
        h=50,
        lr=1e-12,
        epsilon=1.0,
        epsilon_decay=0.99,
        discount=0.0,
    ):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)
        self.connections_info = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.connections_info.append(in_node['Connections'])

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, h),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h, h),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h, POSSIBLE_ACTIONS[0] * POSSIBLE_ACTIONS[1])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def convertNodesToConnections(self, observation, actions):
        unit_group_info = observation[45:106:5]
        converted_actions = np.zeros(actions.shape)
        reward_shaping = []
        for action_num, action in enumerate(actions):
            unit_group_num = int(action[0])
            node_location = int(action[1])
            connections = self.connections_info[int(unit_group_info[unit_group_num])-1]
            action_set = False
            for connection_num, connection in enumerate(connections):
                if connection['ConnectedID'] == node_location:
                    converted_actions[action_num] = np.array((
                        unit_group_num,
                        connection_num
                    ))
                    action_set = True
                    reward_shaping.append(0.0)
                    break
            if not action_set:
                converted_actions[action_num] = np.array((
                    unit_group_num,
                    random.choice(range(len(connections), MAX_NUM_CONNECTIONS))
                ))
                reward_shaping.append(0.0)
        return converted_actions, reward_shaping
    
    def convertConnectionsToNodes(self, observation, actions):
        unit_group_info = observation[45:106:5]
        converted_actions = np.zeros(actions.shape)
        for action_num, action in enumerate(actions):
            unit_group_num = int(action[0])
            connection_num = int(action[1])
            connections = self.connections_info[int(unit_group_info[unit_group_num])-1]
            if len(connections) <= connection_num:
                converted_actions[action_num] = np.array((
                    unit_group_num,
                    unit_group_info[unit_group_num] # This may be 0 or 1 - indexed (likely 1-indexed)
                ))
            else:
                converted_actions[action_num] = np.array((
                    unit_group_num,
                    connections[connection_num]['ConnectedID']
                ))
        return converted_actions

    def filterObservationData(self, observation):
        #return observation
        simplifiedObservation = []
        simplifiedObservation.extend(observation[3:45:4]) # Node percent controlled
        simplifiedObservation.extend(observation[4:45:4]) # Num opponent units
        simplifiedObservation.extend(observation[45:106:5]) # Unit Node Location
        simplifiedObservation.extend(observation[47:106:5]) # Unit Avg Health
        simplifiedObservation.extend(observation[48:106:5]) # Unit In Transit
        return np.array(simplifiedObservation)

    def filterQValuesByValid(self, observation, q_values):
        return q_values
        unit_group_info = observation[45:106:5]
        for unit_num, unit_q_values in enumerate(q_values):
            connections = self.connections_info[int(unit_group_info[unit_num])-1]
            invalid = [i for i in range(11)]
            for connection in connections:
                invalid.remove(connection['ConnectedID']-1)
            invalid.remove(int(unit_group_info[unit_num])-1)
            q_values[unit_num][invalid] = -1e8
        return q_values

    def updateModel(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        y = torch.flatten(y)
        loss = self.criterion(y_pred, torch.Tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            result = self.model(torch.Tensor(state))
            return torch.reshape(result, POSSIBLE_ACTIONS)
        
    def get_greedy_action(self, state):
        filtered_state = self.filterObservationData(state)
        q_values = self.predict(filtered_state).numpy()
        q_values = self.filterQValuesByValid(state, q_values)
        best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
        best_actions_per_group = np.zeros(NUM_UNIT_GROUPS)
        for num, unit_group_q_values in enumerate(q_values):
            best_values_per_group[num] = np.amax(unit_group_q_values)
            best_actions_per_group[num] = np.argmax(unit_group_q_values)
    
        top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]

        actions = np.array([
            [top_n_index, best_actions_per_group[top_n_index]+1] for top_n_index in top_n
        ])
        #print(q_values)
        #print(actions)
        #print('\n')

        return actions

    def get_random_action(self):
        action = np.zeros((7, 2))
        action[:, 0] = np.random.choice(NUM_UNIT_GROUPS, NUM_ACTIONS, replace=False)
        action[:, 1] = np.random.choice([i for i in range(1,11)], NUM_ACTIONS, replace=False)
        return action

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.get_random_action()
        else:
            return self.get_greedy_action(state)
    
    def train(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
    ):
        filtered_previous_state = self.filterObservationData(previous_state)
        q_values = self.predict(filtered_previous_state)
        filtered_q_values = self.filterQValuesByValid(previous_state, q_values)
        for action_num, action in enumerate(actions):
            unit_group_num = int(action[0])
            node_num = int(action[1])
            if len(next_state) == 0:
                q_values[unit_group_num][node_num-1] = reward
            else:
                filtered_next_state = self.filterObservationData(next_state)
                best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
                next_q_values = self.predict(filtered_next_state)
                next_q_values = self.filterQValuesByValid(next_state, next_q_values)
                for num, unit_group_q_values in enumerate(next_q_values):
                    best_values_per_group[num] = np.amax(unit_group_q_values.numpy())
            
                top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]
                maxRewardAvg = np.mean(best_values_per_group[top_n])

                node_control_info = next_state[3:45:4]
                control_reward = np.average(node_control_info) / 100.0

                q_values[unit_group_num][node_num-1] = reward + self.discount * (maxRewardAvg)
        self.updateModel(filtered_previous_state, q_values) 
        time.sleep(0.0)

    def endOfEpisode(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.0)
        print(self.epsilon)