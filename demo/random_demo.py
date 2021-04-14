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
from statsmodels.stats.proportion import proportion_confint

import numpy as np

import utils.reward_shaping as reward_shaping
from utils.Statistics import AgentStatistics

from everglades_server import server
from agents.Smart_State.DQNAgent import DQNAgent
from agents.State_Machine.random_actions_delay import random_actions_delay
from agents.State_Machine.random_actions import random_actions
from agents.State_Machine.base_rush_v1 import base_rushV1
from agents.State_Machine.cycle_target_node11P2 import cycle_targetedNode11P2

from agents.Smart_State.render_smart_state import render_charts

DISPLAY = True # Set whether the visualizer should ever run
TRAIN = False # Set whether the agent should learn or not
RENDER_CHARTS = False # Set whether the training charts should be displayed after training

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
players[0] = random_actions(env.num_actions_per_turn, 0, map_name)
names[0] = 'Random Agent'
players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 20000

#########################
# Statistic variables   #
#########################
k = 100
stats = AgentStatistics(names[0], n_episodes, k, save_file='./saved-stats/smart_state_newton')
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates

ties = 0
losses = 0
score = 0

current_eps = 0
current_loss = 0
q_values = 0

reward = {0: 0, 1: 0}

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

    turn_num = 0
    # Reset the reward average
    while not done:
        if DISPLAY:
            if i_episode % 1 == 0:
                env.render()

        # Get actions for agent
        actions[0] = players[0].get_action( observations[0] )
        # Get actions for random agent
        actions[1] = players[1].get_action( observations[1] )

        # Update env
        observations, reward, done, info = env.step(actions)

        # Increment the turn number
        turn_num += 1


    ################################
    # End of episode agent updates #
    ################################

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
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    ################################
    env.close()