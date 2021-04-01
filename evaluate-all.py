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
from utils.send_imessage import send_imessage

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

PLAYER_1_AGENT_SAVE_FILE = '66-93'

SIGNIFICANCE_LEVEL = 0.95

state_machines_to_test = [
    'Random Actions Delay',
    'Pure Random Actions',
    'Base Rush V1',
    'Cycle Base Rush Turn 25',
    'Cycle Base Rush Turn 50',
    'Cycle Target Node',
    'Cycle Targeted Node 11 Player 2',
    'Swarm Agent'
]

n_episodes = 1000

#########################################

PLAYER_1_AGENT = AgentLoader.loadAgent(
    save_file_path='saved-agents/' + PLAYER_1_AGENT_SAVE_FILE,
    player_num=1
)

##################################################
# LOAD ALL STATE MACHINES THAT WE'D LIKE TO TEST #
##################################################

STATE_MACHINE_AGENT_INFO = []

for state_machine_name in state_machines_to_test:
    STATE_MACHINE_AGENT_INFO.append({
        'Agent Name': state_machine_name,
        'Loaded Agent': loadStateMachine(
            selection=state_machine_name,
            player_num=2
        ),
        'Games Played': 0,
        'Games Beaten': 0
    })

##################################################

## Main Script
players = {}
names = {}

# Set up agents
players[0] = PLAYER_1_AGENT
names[0] = '-'

#################

actions = {}

#######################
#   Evaluation Loop   #
#######################
for i_episode in range(1, n_episodes+1):

    for state_machine in STATE_MACHINE_AGENT_INFO:
        #################################
        # Load the state machine player #
        #################################
        players[1] = state_machine['Loaded Agent']
        names[1] = '-'
        state_machine['Games Played'] += 1

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

            # Get actions for each player
            for pid in players:
                actions[pid] = players[pid].get_action( observations[pid] )

            # Update env
            observations, reward, done, info = env.step(actions)

        ### Updated win calculator to reflect new reward system
        if(reward[0] > reward[1]):
            state_machine['Games Beaten'] += 1
        ###

        env.close()

    ###################
    # Print Progress  #
    ###################
    print('\rEpisode: {}'.format(i_episode))
    ################################

    ########################################
    # Send text notification at milestones #
    ########################################
    if i_episode % 100 == 0:
        send_imessage(
            'Episode {} / {}'.format(i_episode, n_episodes),
            '7246148499'
        )

    env.close()

#######################
# Print Final Results #
#######################

win_rates = []
lower_bounds = []
upper_bounds = []
agent_names = []

print('\n')
for state_machine in STATE_MACHINE_AGENT_INFO:
    confint = proportion_confint(state_machine['Games Beaten'], state_machine['Games Played'], 1.0-SIGNIFICANCE_LEVEL, 'normal')
    confint_range = (confint[1] - confint[0]) * 100.0
    wr = state_machine['Games Beaten'] / state_machine['Games Played'] * 100.0
    upper = confint[1] * 100.0
    lower = confint[0] * 100.0
    print('\rAgent: {}\tWR: {:.2f}%\tActual WR: {:.1f}% ± {:.1f}%\t\tLower: {:.1f}%\tUpper: {:.1f}%'.format(
        state_machine['Agent Name'],
        wr,
        wr,
        confint_range,
        lower,
        upper
    ))

    win_rates.append(wr)
    lower_bounds.append(wr - lower)
    upper_bounds.append(upper - wr)
    agent_names.append(state_machine['Agent Name'])

#####################
# Plot final charts #
#####################

plt.title('Win Rates for ' + PLAYER_1_AGENT_SAVE_FILE + ' @ ' + str(SIGNIFICANCE_LEVEL) + ' Significance Level')

plt.ylabel('Win Rate')
plt.xlabel('Opposing Agent')

plt.bar(
    np.arange(len(win_rates)),
    win_rates,
    yerr=[lower_bounds, upper_bounds],
    capsize=7,
)

plt.xticks(
    np.arange(len(win_rates)),
    agent_names,
    fontsize='small'
)

plt.yticks(
    [0, 20, 40, 60, 75, 80, 95, 100],
    ['0%', '20%', '40%', '60%', '75%', '80%', '95%', '100%']
)

plt.hlines(
    [60],
    -1, 8,
    colors='r',
    linestyles='dashed',
    label='50%'
)
plt.hlines(
    [75],
    -1, 8,
    colors='r',
    linestyles='dashed',
    label='65%'
)
plt.hlines(
    [95],
    -1, 8,
    colors='r',
    linestyles='dashed',
    label='50%'
)

plt.xlim(-0.5, 7.5)
plt.ylim(0, 100)

plt.show()

#######################

######################################################
# Send iMessage to notify that the run has completed #
######################################################
send_imessage('Run Completed!', '7246148499')