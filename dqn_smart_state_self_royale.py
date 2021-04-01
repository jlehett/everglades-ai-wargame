## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque
from statsmodels.stats.proportion import proportion_confint
import random

import numpy as np

import utils.reward_shaping as reward_shaping

from everglades_server import server
from agents.Smart_State.DQNAgent import DQNAgent
from agents.State_Machine.random_actions_delay import random_actions_delay
from agents.State_Machine.random_actions import random_actions

DISPLAY = False # Set whether the visualizer should ever run
TRAIN_TEAM_0 = True # Set whether the agents on team 0 should learn or not
TRAIN_TEAM_1 = True # Set whether the agents on team 1 should learn or not

NUM_AGENTS_PER_TEAM = 4

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
team0 = []
team1 = []
for i in range(1, NUM_AGENTS_PER_TEAM+1):
    team0.append(
        DQNAgent(
            player_num=0,
            map_name=map_name,
            train=TRAIN_TEAM_0,
            network_save_name='agents/Smart_State/saved_models/self-player-0-' + str(i),
            network_load_name='agents/Smart_State/saved_models/self-player-0-' + str(i),
        )
    )
    team1.append(
        DQNAgent(
            player_num=1,
            map_name=map_name,
            train=TRAIN_TEAM_1,
            network_save_name='agents/Smart_State/saved_models/self-player-1-' + str(i),
            network_load_name='agents/Smart_State/saved_models/self-player-1-' + str(i)
        )
    )

names[0] = 'DQN Agent - Team 0'
names[1] = 'DQN Agent - Team 1'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 5000

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
    # Pick random agents from each team to compete
    players[0] = random.choice(team0)
    players[1] = random.choice(team1)

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
    turn_num = 0

    # Reset the reward average
    while not done:
        if DISPLAY:
            if i_episode % 5 == 0:
                env.render()

        # Create storage for previous observations and directions for each player
        prev_observations = [None, None]
        directions = [None, None]

        # Get actions for each player
        for pid in players:
            actions[pid], directions[pid] = players[pid].get_action( observations[pid] )

        # Grab previous observations for each agent
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
            directions[0],
            reward_shaping.transition(
                reward_shaping.normalized_score,
                reward_shaping.reward_short_games,
                1000,
                i_episode,
                0,
                reward,
                done,
                turn_num
            )
        )
        players[0].optimize_model()

        players[1].remember_game_state(
            prev_observations[1],
            observations[1],
            directions[1],
            reward_shaping.transition(
                reward_shaping.normalized_score,
                reward_shaping.reward_short_games,
                1000,
                i_episode,
                1,
                reward,
                done,
                turn_num
            )
        )
        players[1].optimize_model()
        #########################

        current_eps = players[0].epsilon

        # Increment the turn number
        turn_num += 1


    ################################
    # End of episode agent updates #
    ################################
    for agent in team0:
        if agent == players[0]:
            agent.end_of_episode(i_episode)
        else:
            agent.end_of_episode_not_play(i_episode)
    for agent in team1:
        if agent == players[1]:
            agent.end_of_episode(i_episode)
        else:
            agent.end_of_episode_not_play(i_episode)

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
    if TRAIN_TEAM_0 or TRAIN_TEAM_1:
        print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {}\tEpsilon: {:.2f}\tLR: {:.2e}\tTies: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses,current_eps, players[0].learning_rate, ties), end="")
        if i_episode % k == 0:
            print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
            short_term_scores.append(np.mean(short_term_wr))
            short_term_wr = np.zeros((k,), dtype=int)
    else:
        confint = proportion_confint(score, i_episode, 0.05, 'normal')
        confint_range = (confint[1] - confint[0]) * 100.0
        if i_episode > 50:
            print('\rEpisode: {}\tCurrent WR: {:.2f}%\tActual WR: {:2.1f}% ± {:2.1f}%\t\tLower: {:2.1f}%\tUpper: {:2.1f}%'.format(i_episode, current_wr * 100.0, current_wr * 100.0, confint_range / 2.0, confint[0]*100.0, confint[1]*100.0))
        else:
            print('\rEpisode: {}\tCurrent WR: {:.2f}%\tNot Enough Data to Determine Actual WR'.format(i_episode, current_wr * 100.0))
        
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

#########