## Static Imports
import os, sys
sys.path.insert(0, '.')

import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque
import random

import numpy as np

from everglades_server import server
from agents.Minimized.DQNAgent import DQNAgent
from agents.State_Machine.random_actions_delay import random_actions_delay
from agents.State_Machine.base_rush_v1 import base_rushV1

DISPLAY = True # Set whether the visualizer should ever run
TRAIN = False # Set whether the agent should learn or not

#############################
# Environment Config Setup  #
#############################
map_name = "DemoMap.json"
config_dir = './config/'  
map_file = config_dir + map_name
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'
#############################

debug = False

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

#################
# Setup agents  #
#################
players[0] = DQNAgent(
    player_num=0,
    map_name=map_name,
<<<<<<< HEAD:dqn_training.py
    )
names[0] = "DQN Agent"
players[1] = base_rushV1(env.num_actions_per_turn, 1)
names[1] = 'Random Agent Delay'
=======
    train=True,
    network_save_name='./agents/Minimized/saved_models/self-player-0',
    network_load_name='./agents/Minimized/saved_models/self-player-0',
)
names[0] = "DQN Agent - Player 0"
players[1] = random_actions_delay(env.num_actions_per_turn, 1, map_name)
names[1] = "Opposing Agent"
>>>>>>> origin:agents/Minimized/training_scripts/dqn_training.py
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 2

#########################
# Statistic variables   #
#########################
scores = []
k = 100
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates
short_term_scores = [0.5] # Average win rates per k episodes
ties = 0
losses = 0
score = 0
current_eps = 0

epsilonVals = []
current_loss = 0
lossVals = []
average_reward = 0
avgRewardVals = []

#####################
#   Training Loop   #
#####################
for i_episode in range(1, n_episodes+1):
    #################
    #   Game Loop   #
    #################
    done = 0
    observations = env.reset(
        players=players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = output_dir,
        pnames = names,
        debug = debug
    )


    # Reset the reward average
    while not done:
<<<<<<< HEAD:dqn_training.py
        if DISPLAY and i_episode % 5 == 0:
            env.render()
=======
        # if i_episode % 5 == 0:
        #     env.render()
>>>>>>> origin:agents/Minimized/training_scripts/dqn_training.py

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Grab previous observation for agent
        prev_observations = [None, None]
        prev_observations[0] = observations[0]
        prev_observations[1] = observations[1]

        # Update env
        observations, reward, done, info = env.step(actions)

        #########################
        # Handle agent update   #
        #########################
        players[0].remember_game_state(
            prev_observations[0],
            observations[0],
            actions[0],
            reward[0]
        )
        players[0].optimize_model()

        if selected_opponent['type'] == 'dqn':
            players[1].remember_game_state(
                prev_observations[1],
                observations[1],
                actions[1],
                reward[1]
            )
            players[1].optimize_model()
        #########################

        current_eps = players[0].epsilon


    ################################
    # End of episode agent updates #
    ################################
    players[0].end_of_episode(i_episode)

    ### Updated win calculator to reflect new reward system
    if(reward[0] > reward[1]):
        score += 1
        short_term_wr[(i_episode-1)%k] = 1
    elif(reward[0] == reward[1]):
        ties += 1
    else:
        losses += 1
    ###

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    scores.append(score / i_episode) ## save the most recent score
    current_wr = score / i_episode
    epsilonVals.append(current_eps)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
<<<<<<< HEAD:dqn_training.py
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {}\tEpsilon: {:.2f}\tTies: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses,current_eps, ties), end="")
=======
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {}\tEpsilon: {:.2f}\tLR: {:.2e}\tTies: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses,current_eps, players[0].learning_rate, ties), end="")
>>>>>>> origin:agents/Minimized/training_scripts/dqn_training.py
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

    ################################
    env.close()

#####################
# Plot final charts #
#####################
fig, (ax1, ax2) = plt.subplots(2)

#########################
#   Epsilon Plotting    #
#########################
par1 = ax1.twinx()
par2 = ax2.twinx()
#########################

######################
#   Cumulative Plot  #
######################
ax1.set_ylim([0.0,1.0])
fig.suptitle('Win rates')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative win rate')
ax1.yaxis.label.set_color('blue')
par1.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
par1.set_ylabel('Epsilon')
par1.yaxis.label.set_color('green')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
ax2.set_ylim([0.0,1.0])
par2.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
par2.set_ylabel('Epsilon')
par2.yaxis.label.set_color('green')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average win rate')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')
plt.show()
#############################
