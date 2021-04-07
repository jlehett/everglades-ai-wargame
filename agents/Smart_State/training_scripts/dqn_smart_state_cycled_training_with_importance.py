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
from statsmodels.stats.proportion import proportion_confint

import numpy as np

import utils.reward_shaping as reward_shaping

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

IMPORTANCE_UPDATE_AFTER = 50

DISPLAY = False # Set whether the visualizer should ever run
TRAIN = True # Set whether the agent should learn or not

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
    network_load_name=None,
)
names[0] = "DQN Agent"

# Create an array of all agents that could be used during training
opposing_agents = [
    {
        'name': 'Random Agent Delay',
        'agent': random_actions_delay(env.num_actions_per_turn, 1, map_name),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Random Agent',
        'agent': random_actions(env.num_actions_per_turn, 1, map_name),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Bull Rush',
        'agent': bull_rush(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'All Cycle',
        'agent': all_cycle(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Base Rush v1',
        'agent': base_rushV1(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Rush Turn 25',
        'agent': Cycle_BRush_Turn25(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Rush Turn 50',
        'agent': Cycle_BRush_Turn50(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Target Node',
        'agent': Cycle_Target_Node(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Targeted Node 1',
        'agent': cycle_targetedNode1(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Targeted Node 11',
        'agent': cycle_targetedNode11(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Cycle Target Node 11 P2',
        'agent': cycle_targetedNode11P2(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Random Actions 2',
        'agent': random_actions_2(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Same Commands 2',
        'agent': same_commands_2(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Same Commands',
        'agent': same_commands(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    },
    {
        'name': 'Swarm Agent',
        'agent': SwarmAgent(env.num_actions_per_turn, 1),
        'games': 0,
        'wins': 0,
    }
]

# Create importance weights for all of the opposing agents
opposing_agent_weights = [1.0 for i in range(len(opposing_agents))]

# Define a function to update the weights based on win rate
def updateAgentWeights():
    opposing_agent_weights = []
    for index, opposing_agent in enumerate(opposing_agents):
        if opposing_agent['games'] == 0:
            opposing_agent_weights.append(1.0)
        else:
            opposing_agent_weights.append(1.0 - opposing_agent['wins'] / opposing_agent['games'] + 0.05)
    return opposing_agent_weights

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
    # Determine the opposing agent to play against
    episode_opposing_agent = random.choices(opposing_agents, opposing_agent_weights)[0]

    # Set the opposing agent for the episode
    players[1] = episode_opposing_agent['agent']
    names[1] = episode_opposing_agent['name']

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

        # Get actions for agent
        actions[0], directions = players[0].get_action(observations[0])
        # Get actions for state machine player 2
        actions[1] = players[1].get_action(observations[1])

        # Grab previos observation for agent
        prev_observation = observations[0]

        # Update env
        observations, reward, done, info = env.step(actions)

        #########################
        # Handle agent update   #
        #########################
        players[0].remember_game_state(
            prev_observation,
            observations[0],
            directions,
            reward_shaping.transition(
                reward_shaping.normalized_score,
                reward_shaping.reward_short_games,
                200,
                i_episode,
                0,
                reward,
                done,
                turn_num
            )
        )
        players[0].optimize_model()
        #########################

        current_eps = players[0].epsilon

        # Increment the turn number
        turn_num += 1


    ################################
    # End of episode agent updates #
    ################################
    players[0].end_of_episode(i_episode)

    ### Updated win calculator to reflect new reward system
    episode_opposing_agent['games'] += 1
    if(reward[0] > reward[1]):
        score += 1
        short_term_wr[(i_episode-1)%k] = 1
        episode_opposing_agent['wins'] += 1
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
    if TRAIN:
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
        
    # Update the opposing agent weights if appropriate
    if i_episode % IMPORTANCE_UPDATE_AFTER == 0:
        print('Updating Opposing Agent Weights...')
        opposing_agent_weights = updateAgentWeights()
        print('Opposing Agent Weights:', opposing_agent_weights)
        
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