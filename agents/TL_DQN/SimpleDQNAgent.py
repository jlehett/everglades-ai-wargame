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

# Reward shaping toggles
GAME_LOST_REWARD_SHAPING_TOGGLE = True

# Reward shaping constants
GAME_LOST_REWARD_SHAPING = -1
GAME_WON_REWARD_SHAPING = 1

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
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, POSSIBLE_ACTIONS[0] * POSSIBLE_ACTIONS[1])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    """
        Function for preprocessing the state such that the values are mostly
        normalized to the range -1.0 to 1.0.
    """
    def preprocessState(self, state):
        newState = np.copy(state)
        newState[0] /= 151.0
        newState[3:45:4] /= 100.0
        newState[4:45:4] /= 100.0
        newState[45:106:5] /= 11.0
        newState[46:106:5] /= 3.0
        newState[47:106:5] /= 100.0
        newState[49:106:5] /= 12.0
        return newState

    """
        Function for training the model given a state and the target Q values
    """
    def updateModel(self, state, y):
        # Pre-process the state first
        state = self.preprocessState(state)
        # Get the predicted Q values given the state
        y_pred = self.model(torch.Tensor(state))
        y = torch.flatten(y)
        # Compute the loss between the predicted Q values and the target Q values.
        loss = self.criterion(y_pred, torch.Tensor(y))
        # Have the network train with the predicted and target Q values.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    """
        Function for predicting the Q values of all actions for a given state.
    """
    def predict(self, state):
        # Pre-process the state first
        state = self.preprocessState(state)
        # torch.no_grad() tells the network not to keep the results of this
        # prediction in its memory for future training
        with torch.no_grad():
            result = self.model(torch.Tensor(state))
            return torch.reshape(result, POSSIBLE_ACTIONS)
    
    """
        Function for getting the greedy action via the network predicting
        the Q values and choosing the best 7 Q values of the prediction
        (without repeating unit groups).
    """
    def get_greedy_action(self, state):
        # Get the predicted Q values for the state via the network
        q_values = self.predict(state).numpy()
        # This code is just to find the top 7 actions predicted by the network.
        best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
        best_actions_per_group = np.zeros(NUM_UNIT_GROUPS)
        for num, unit_group_q_values in enumerate(q_values):
            best_values_per_group[num] = np.amax(unit_group_q_values)
            best_actions_per_group[num] = np.argmax(unit_group_q_values)
    
        top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]
        # Put the top 7 actions into a form readable by the game.
        actions = np.array([
            [top_n_index, best_actions_per_group[top_n_index]+1] for top_n_index in top_n
        ])

        return actions

    """
        Function for getting a completely random action for the Everglades
        environment. Can include repeats and illegal actions.
    """
    def get_random_action(self):
        action = np.zeros((7, 2))
        action[:, 0] = np.random.choice(NUM_UNIT_GROUPS, NUM_ACTIONS, replace=False)
        action[:, 1] = np.random.choice([i for i in range(1,11)], NUM_ACTIONS, replace=False)
        return action

    """
        Function for getting an action to pass back to the Everglades environment.
        Can be either greedy (exploitation) or random (exploration) depending on
        the current epsilon value for the agent.
    """
    def get_action(self, state):
        # Exploration phase (random action)
        if random.random() < self.epsilon:
            return self.get_random_action()
        # Exploitation phase (greedy action)
        else:
            return self.get_greedy_action(state)
    
    """
        Function to get the modified reward for the agent to use during training.
        Different reward shaping schemes can be set up via the constants declared
        at the top of this file.
    """
    def get_reward_shaping_modifier(self, next_state, reward):
        # total_reward will track the final reward to use in training on this step.
        total_reward = 0
        # Modify the game won reward if it was registered
        if reward == 1:
            total_reward += GAME_WON_REWARD_SHAPING
        # Add reward shaping if the game has just been lost
        if len(next_state) == 0:
            if reward != 1 and GAME_LOST_REWARD_SHAPING_TOGGLE:
                total_reward += GAME_LOST_REWARD_SHAPING
        # Return the final reward
        return total_reward
            
    """
        Function to train the network given the previous game state, the
        actions the agent took, the state the agent is in after taking the
        specified actions, and the reward the Everglades environment sent
        to the agent.
    """
    def train(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
    ):
        # Grab the old predicted Q values for the previous state
        q_values = self.predict(previous_state)
        # Perform reward shaping
        shaped_reward = self.get_reward_shaping_modifier(next_state, reward)
        # The target for the network to train on should be the previous predicted
        # Q values, with the rewards for the actions taken changed to the actual
        # reward felt by the agent for taking the action + the expected future
        # reward.
        for action_num, action in enumerate(actions):
            unit_group_num = int(action[0])
            node_num = int(action[1])
            # In this state, the agent has finished the game. We do not need to
            # add the expected future reward since there won't be a future reward.
            if len(next_state) == 0:
                q_values[unit_group_num][node_num-1] = shaped_reward
            # In this state, the agent still has not finished the game. We need
            # to predict the future expected reward given the next state and add
            # that to the reward felt by the agent for taking the last set of actions.
            else:
                # This code is just to find the top 7 actions predicted by the network.
                best_values_per_group = np.zeros(NUM_UNIT_GROUPS)
                next_q_values = self.predict(next_state)
                for num, unit_group_q_values in enumerate(next_q_values):
                    best_values_per_group[num] = np.amax(unit_group_q_values.numpy())
            
                top_n = np.argpartition(best_values_per_group, -NUM_ACTIONS)[-NUM_ACTIONS:]
                # Compute the average reward for the top 7 actions to use for the
                # discounted reward
                maxRewardAvg = np.mean(best_values_per_group[top_n])
                # Change the Q values as mentioned above for training.
                q_values[unit_group_num][node_num-1] = shaped_reward + self.discount * (maxRewardAvg)
        # Train the network given the actual Q values felt by the network.
        self.updateModel(previous_state, q_values)

    """
        Function to handle some clean up after a full game has been played
        by the agent.

        Currently, only decreasing the epsilon value at the end of each game.
    """
    def endOfEpisode(self):
        print(self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.0)