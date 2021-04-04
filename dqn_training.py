## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque
import torch

import numpy as np

from everglades_server import server
from agents.DQN.DQNAgent import DQNAgent
from agents.State_Machine.random_actions import random_actions
from agents.State_Machine.random_actions_delay import random_actions_delay
from utils.reward_shaping import *
from utils.Statistics import AgentStatistics

from agents.DQN.render_dqn import render_charts

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

#########################
#   Setup DQN Constants #
#########################
LR = 0.0001
REPLAY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.00001
TARGET_UPDATE = 100 # 60 games approximately equal to ~10,000 time steps (what was used by Deepmind)
NETWORK_SAVE_NAME = "saved-agents/dqn_new"
SAVE_AFTER_EPISODE = 100
DEVICE = "CPU"
#########################

DISPLAY = False
TRAIN = True
RENDER_CHARTS = True

#################
# Setup agents  #
#################
players[0] = DQNAgent(env.num_actions_per_turn, 
                        env.observation_space,
                        0,
                        LR,
                        REPLAY_SIZE,
                        BATCH_SIZE,
                        GAMMA,
                        EPS_START,
                        EPS_END,
                        EPS_DECAY,
                        TARGET_UPDATE, 
                        DEVICE, 
                        TRAIN, 
                        SAVE_AFTER_EPISODE, 
                        NETWORK_SAVE_NAME)
names[0] = "DQN Agent"
players[1] = random_actions_delay(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent Delay'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 5000

#########################
# Statistic variables   #
#########################
k = 100
stats = AgentStatistics(names[0], n_episodes, k, save_file="saved-stats/dqn_new")
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates

ties = 0
losses = 0
score = 0

current_eps = 0
current_loss = 0
q_values = 0

reward = {0: 0, 1: 0}
#########################

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

    turnNum = 0
    while not done:
        if DISPLAY and i_episode % 5 == 0:
            env.render()

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Grab previos observation for agent
        prev_observation = observations[0]

        # Update env
        observations, reward, done, info = env.step(actions)

        #########################
        # Handle agent update   #
        #########################
        turn_scores = reward_short_games(0, reward, done, turnNum)

        players[0].remember_game_state(prev_observation, actions[0], turn_scores, observations[0])

        # Handle end of game updates
        if done:
            players[0].end_of_episode(i_episode)
        
        players[0].optimize_model()
        #########################

        current_eps = players[0].eps_threshold
        q_values += players[0].q_values.mean()
        current_loss = players[0].loss

        turnNum += 1
    #####################
    #   End Game Loop   #
    #####################

    if(reward[0] > reward[1]):
        score += 1
        short_term_wr[(i_episode-1)%k] = 1
    elif(reward[0] == reward[1]):
        ties += 1
    else:
        losses += 1

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    stats.scores.append(score / i_episode) ## save the most recent score
    current_wr = score / i_episode
    stats.epsilons.append(current_eps)
    stats.network_loss.append(current_loss)
    q_values = q_values / 150
    stats.q_values.append(q_values)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f} Wins: {} Losses: {} Ties: {} Eps: {:.2f} Loss: {:.2f} Average Q-Value: {:.2f}\n'.format(i_episode,current_wr,score,losses,ties,current_eps, current_loss,q_values), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        stats.short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

    ################################
    env.close()
    #########################
    #   End Training Loop   #
    #########################

# Save final model state
players[0].save_network(i_episode)

# Render charts to show visual of training stats
if RENDER_CHARTS:
    render_charts(stats)

# Save run stats
stats.save_stats()
