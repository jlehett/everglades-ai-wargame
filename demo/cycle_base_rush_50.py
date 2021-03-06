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
from agents.State_Machine.bull_rush import bull_rush
from agents.State_Machine.all_cycle import all_cycle
from agents.State_Machine.base_rush_v1 import base_rushV1
from agents.State_Machine.cycle_rush_turn25 import Cycle_BRush_Turn25
from agents.State_Machine.cycle_rush_turn50 import Cycle_BRush_Turn50
from agents.State_Machine.cycle_target_node import Cycle_Target_Node
from agents.State_Machine.cycle_target_node1 import cycle_targetedNode1
from agents.State_Machine.cycle_target_node11 import cycle_targetedNode11
from agents.State_Machine.cycle_target_node11P2 import cycle_targetedNode11P2
from agents.State_Machine.random_actions_2 import random_actions_2
from agents.State_Machine.same_commands_2 import same_commands_2
from agents.State_Machine.same_commands import same_commands
from agents.State_Machine.swarm_agent import SwarmAgent

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
players[0] = DQNAgent(
    player_num=0,
    map_name=map_name,
    train=TRAIN,
    network_save_name=None,
    network_load_name='./saved-agents/best_smart_state',
)
names[0] = "DQN Agent"
players[1] = Cycle_BRush_Turn50(env.num_actions_per_turn, 1)
names[1] = 'Cycle Rush Turn 50'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 50

#########################
# Statistic variables   #
#########################
k = 100
stats = AgentStatistics(names[0], n_episodes, k, save_file='/saved-stats/local')
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
            env.render()

        # Get actions for agent
        actions[0], directions = players[0].get_action( observations[0] )
        # Get actions for random agent
        actions[1] = players[1].get_action( observations[1] )

        # Grab previos observation for agent
        prev_observation = observations[0]

        # Update env
        observations, reward, done, info = env.step(actions)

        #########################
        # Handle agent update   #
        #########################
        turn_scores = reward_shaping.reward_short_games(0, reward, done, turn_num)

        players[0].remember_game_state(
            prev_observation,
            observations[0],
            directions,
            turn_scores
        )
        players[0].optimize_model()
        #########################

        current_eps = players[0].epsilon
        current_loss = players[0].loss

        # Increment the turn number
        turn_num += 1


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
    if TRAIN:
        stats.scores.append(score / i_episode)
        current_Wr = score / i_episode
        stats.epsilons.append(current_eps)
        stats.network_loss.append(current_loss)
        q_values = 0
        stats.q_values.append(q_values)

    current_wr = score / i_episode
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    if TRAIN:
        print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {}\tEpsilon: {:.2f}\tLR: {:.2e}\tTies: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses,current_eps, players[0].learning_rate, ties), end="")
        if i_episode % k == 0:
            print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
            stats.short_term_scores.append(np.mean(short_term_wr))
            short_term_wr = np.zeros((k,), dtype=int)
    else:
        confint = proportion_confint(score, i_episode, 0.05, 'normal')
        confint_range = (confint[1] - confint[0]) * 100.0
        print('\rEpisode: {}\tCurrent WR: {:.2f}%\tLower: {:2.1f}%\tUpper: {:2.1f}%'.format(i_episode, current_wr * 100.0, confint[0]*100.0, confint[1]*100.0))
    ################################
    env.close()

# Save final model state
players[0].save_network(i_episode)

# Render charts to show visual of training stats
if RENDER_CHARTS:
    render_charts(stats)

# Save run stats
stats.save_stats()