from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape
from keras.models import Model
from collections import namedtuple, deque
import numpy as np
import random

INPUT_SPACE = 105
NUM_UNIT_GROUPS = 12
NUM_NODES = 11
NUM_ACTIONS = 7
ACTION_SPACE = NUM_NODES*NUM_UNIT_GROUPS

class KerasDQNAgent:
    def __init__(
        self,
        lr=0.01,
        discount=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.0,
        batch_size=64,
        train_start=500,
        memory_size=10000,
    ):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        self.memory_size = memory_size

        self.action_choices = self.get_action_choices(
            (NUM_UNIT_GROUPS * NUM_NODES, 2)
        )
        self.memory = ReplayMemory(memory_size)
        self.model = self.constructModel()

    def constructModel(self):
        X_input = Input((INPUT_SPACE,))

        X = Dense(512, input_shape=((INPUT_SPACE,)), activation='relu', kernel_initializer='he_uniform')(X_input)
        X = Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
        X = Dense(ACTION_SPACE, activation='linear', kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='Everglades-DQN')
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])

        model.summary()
        return model
    
    def train(
        self,
        previous_state=None,
        next_state=[],
        actions=None,
        reward=None,
        done=False,
    ):
        if next_state == []:
            next_state = [0 for i in range(105)]
        self.memory.push(previous_state, actions, next_state, reward, done)

        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        next_state_tensor = np.asarray(batch.next_state)
        state_tensor = np.asarray(batch.state)
        action_tensor = np.asarray(batch.action)
        reward_tensor = np.asarray(batch.reward)
        done_tensor = np.asarray(batch.done)

        next_state_batch = next_state_tensor
        state_batch = state_tensor
        action_batch = action_tensor
        reward_batch = reward_tensor
        done_batch = done_tensor

        target = self.model.predict(state_batch)
        target = np.reshape(target, (target.shape[0], NUM_UNIT_GROUPS, NUM_NODES))
        target_next = self.model.predict(next_state_batch)

        for i in range(self.batch_size):
            if done_batch[i]:
                for action in action_batch[i]:
                    target[i][int(action[0])][int(action[1]-1)] = reward_batch[i]
            else:
                max_next = (-target_next[i]).argsort()[:NUM_ACTIONS]
                for j, action in enumerate(action_batch[i]):
                    target[i][int(action[0])][int(action[1]-1)] = reward_batch[i] + self.discount * target_next[i][max_next[j]]
        
        target = np.reshape(target, (target.shape[0], NUM_UNIT_GROUPS*NUM_NODES))
        self.model.fit(state_batch, target, batch_size=self.batch_size, verbose=0)
    
    def get_action(self, obs):
        if np.random.random_sample() <= self.epsilon:
            return self.get_random_action()
        else:
            return self.get_greedy_action(obs)

    def get_random_action(self):
        action = np.zeros((NUM_ACTIONS, 2))
        action[:, 0] = np.random.choice(NUM_UNIT_GROUPS, NUM_ACTIONS, replace=False)
        action[:, 1] = np.random.choice([i for i in range(1,11)], NUM_ACTIONS, replace=False)
        return action

    def get_greedy_action(self, obs):
        action = np.zeros((NUM_ACTIONS, 2))
        pred = self.model.predict([[i for i in obs]])
        maxIndices = (-pred[0]).argsort()[:NUM_ACTIONS]
        for i in range(0, NUM_ACTIONS):
            action[i] = self.action_choices[maxIndices[i]]
        results = (maxIndices, action)
        return results[1]

    def get_action_choices(self, shape):
        action_choices = np.zeros(shape)
        group_id = 0
        node_id = 1
        for i in range(0, action_choices.shape[0]):
            if i > 0 and i % 11 == 0:
                group_id += 1
                node_id = 1
            action_choices[i] = [group_id, node_id]
            node_id += 1
        return action_choices

    def end_of_episode(self, game_num):
        pass


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)