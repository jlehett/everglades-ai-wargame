import os
import numpy as np
import random
import torch
import torch.nn as nn

OBSERVATION_DIM = 105
NUM_UNIT_GROUPS = 12
NUM_NODES = 11
NUM_ACTIONS = 7
MAX_NUM_CONNECTIONS = 6

POSSIBLE_ACTIONS = (NUM_UNIT_GROUPS, NUM_NODES)

import torch
import random
import numpy as np

class SimpleDQNAgent:
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

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, h),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h, h),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h, POSSIBLE_ACTIONS[0] * POSSIBLE_ACTIONS[1])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

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
        q_values = self.predict(state).numpy()
        best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
        best_actions_per_group = np.zeros(NUM_UNIT_GROUPS)
        for num, unit_group_q_values in enumerate(q_values):
            best_values_per_group[num] = np.amax(unit_group_q_values)
            best_actions_per_group[num] = np.argmax(unit_group_q_values)
    
        top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]

        actions = np.array([
            [top_n_index, best_actions_per_group[top_n_index]+1] for top_n_index in top_n
        ])

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
        q_values = self.predict(previous_state)
        for action_num, action in enumerate(actions):
            unit_group_num = int(action[0])
            node_num = int(action[1])
            if len(next_state) == 0:
                q_values[unit_group_num][node_num-1] = reward
            else:
                best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
                next_q_values = self.predict(next_state)
                for num, unit_group_q_values in enumerate(next_q_values):
                    best_values_per_group[num] = np.amax(unit_group_q_values.numpy())
            
                top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]
                maxRewardAvg = np.mean(best_values_per_group[top_n])

                q_values[unit_group_num][node_num-1] = reward + self.discount * (maxRewardAvg)
        self.updateModel(previous_state, q_values)

    def endOfEpisode(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.0)
        print(self.epsilon)