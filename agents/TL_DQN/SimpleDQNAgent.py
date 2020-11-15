import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

OBSERVATION_DIM = 105
NUM_UNIT_GROUPS = 12
NUM_NODES = 11
NUM_ACTIONS = 7
MAX_NUM_CONNECTIONS = 6

POSSIBLE_ACTIONS = (NUM_UNIT_GROUPS, NUM_NODES)

# Reward shaping toggles
GAME_LOST_REWARD_SHAPING_TOGGLE = False
TERRITORY_CONTROL_REWARD_SHAPING_TOGGLE = False

# Reward shaping constants
GAME_LOST_REWARD_SHAPING = -1
GAME_WON_REWARD_SHAPING = 1
TERRITORY_CONTROL_REWARD_SHAPING_MODIFIER = 1

import torch
import random
import numpy as np

class SimpleDQNAgent:
    def __init__(
        self,
        env,
        map_name,
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
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 132),
            torch.nn.ReLU(),
            torch.nn.Linear(132, POSSIBLE_ACTIONS[0] * POSSIBLE_ACTIONS[1])
        )
        self.optimizer = torch.optim.RMSprop(self.policy_model.parameters())

        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_DIM, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 132),
            torch.nn.ReLU(),
            torch.nn.Linear(132, POSSIBLE_ACTIONS[0] * POSSIBLE_ACTIONS[1])
        )
        self.copyParamsToTargetModel()
        self.target_model.eval()

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
        y_target = self.target_model(torch.Tensor(state))
        y = torch.flatten(y)
        # Compute the loss between the predicted Q values and the target Q values.
        loss = self.criterion(y_target, torch.Tensor(y))
        # Have the network train with the predicted and target Q values.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    """
        Function to copy the parameters from the predicted model to the
        target model.
    """
    def copyParamsToTargetModel(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    """
        Function for predicting the Q values of all actions for a given state.
    """
    def predict(self, state):
        # Pre-process the state first
        state = self.preprocessState(state)
        # torch.no_grad() tells the network not to keep the results of this
        # prediction in its memory for future training
        with torch.no_grad():
            result = self.policy_model(torch.Tensor(state))
            return torch.reshape(result, POSSIBLE_ACTIONS)
    
    """
        Function for predicting the Q values of all actions for a given state
        using the target model.
    """
    def predictUsingTarget(self, state):
        # Pre-process the state first
        state = self.preprocessState(state)
        # torch.no_grad() tells the network not to keep the results of this
        # prediction in its memory for future training
        with torch.no_grad():
            result = self.target_model(torch.Tensor(state))
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
        if GAME_LOST_REWARD_SHAPING_TOGGLE:
            if len(next_state) == 0:
                if reward != 1:
                    total_reward += GAME_LOST_REWARD_SHAPING
        # Add reward shaping for the territories controlled
        if TERRITORY_CONTROL_REWARD_SHAPING_TOGGLE:
            if len(next_state) != 0:
                territory_control_info = next_state[3:45:4] / 100.0
                avg_territory_control = np.average(territory_control_info)
                total_reward += avg_territory_control * TERRITORY_CONTROL_REWARD_SHAPING_MODIFIER
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
        # Perform reward shaping
        shaped_reward = self.get_reward_shaping_modifier(next_state, reward)
        # Grab the state action values given the policy model
        preprocessed_previous_state = self.preprocessState(previous_state)
        state_action_values = self.policy_model(torch.Tensor(preprocessed_previous_state))
        # We need to gather up the 7 actions we actually took
        actionIndices = [int(action[0] * (action[1] -1)) for action in actions]
        state_action_values = state_action_values[actionIndices]
        # We need to predict the next state values. If the game is over
        # the next state values should all be zero.
        next_state_values = torch.zeros(1)
        if len(next_state) != 0:
            next_state_values = self.predictUsingTarget(next_state).max(1).values
        # Set the expected state action values
        print(next_state_values)
        expected_state_action_values = (next_state_values * self.discount) + shaped_reward
        # Compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    """
        Function to handle some clean up after a full game has been played
        by the agent.

        Currently, only decreasing the epsilon value at the end of each game.
    """
    def endOfEpisode(self):
        print(self.epsilon)
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.0)