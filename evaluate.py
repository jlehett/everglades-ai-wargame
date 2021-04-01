## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque 
import random
from statsmodels.stats.proportion import proportion_confint

import numpy as np

from utils.Agent_Loader.agent_loader import AgentLoader as _AgentLoader

from everglades_server import server
from agents.Minimized.DQNAgent import DQNAgent
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

#############################
# Environment Config Setup  #
#############################
map_name = "DemoMap.json"
config_dir = './config/'  
map_file = config_dir + map_name
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

env = gym.make('everglades-v0')
#############################

# Initialize the AgentLoader object
AgentLoader = _AgentLoader(map_name)

# Create a function to load a state machine agent
def loadStateMachine(selection, player_num):
    # Return the selection
    if selection in ['Random Actions Delay', 'Pure Random Actions']:
        return STATE_MACHINE_OPTIONS[selection](env.num_actions_per_turn, player_num, map_name)
    else:
        return STATE_MACHINE_OPTIONS[selection](env.num_actions_per_turn, player_num)

###############################
# STATE MACHINE AGENT OPTIONS #
###############################

STATE_MACHINE_OPTIONS = {
    'Random Actions Delay': random_actions_delay,
    'Pure Random Actions': random_actions,
    'Bull Rush': bull_rush,
    'All Cycle': all_cycle,
    'Base Rush V1': base_rushV1,
    'Cycle Base Rush Turn 25': Cycle_BRush_Turn25,
    'Cycle Base Rush Turn 50': Cycle_BRush_Turn50,
    'Cycle Target Node': Cycle_Target_Node,
    'Cycle Targeted Node 1': cycle_targetedNode1,
    'Cycle Targeted Node 11': cycle_targetedNode11,
    'Cycle Targeted Node 11 Player 2': cycle_targetedNode11P2,
    'Random Actions 2': random_actions_2,
    'Same Commands 2': same_commands_2,
    'Same Commands': same_commands,
    'Swarm Agent': SwarmAgent,
}

#########################################
# SET THESE TO CUSTOMIZE THE EVALUATION #
#########################################

PLAYER_1_AGENT = AgentLoader.loadAgent(
    save_file_path='saved-agents/70-87_fc2',
    player_num=1
)

PLAYER_2_AGENT = loadStateMachine(
    selection='Base Rush V1',
    player_num=2
)

RENDER_EVERY_N_EPISODES = 1

#########################################

## Main Script
players = {}
names = {}

# Set up agents
players[0] = PLAYER_1_AGENT
names[0] = '-'
players[1] = PLAYER_2_AGENT
names[1] = '-'

#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 5000

#########################
# Statistic variables   #
#########################
scores = []
losses = 0
score = 0

#######################
#   Evaluation Loop   #
#######################
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
        debug = False
    )

    while not done:
        if RENDER_EVERY_N_EPISODES != 0 and i_episode % RENDER_EVERY_N_EPISODES == 0:
            env.render()

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Update env
        observations, reward, done, info = env.step(actions)

    ### Updated win calculator to reflect new reward system
    if(reward[0] > reward[1]):
        score += 1
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
    #############################################

    ################################
    # Print evaluation statistics  #
    ################################
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
fig, (ax1) = plt.subplots(1)

######################
#   Cumulative Plot  #
######################
ax1.set_ylim([0.0,1.0])
fig.suptitle('Win rates')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative win rate')
ax1.yaxis.label.set_color('blue')
#######################