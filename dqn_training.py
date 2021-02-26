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
from agents.Minimized.DQNAgent import DQNAgent
from agents.State_Machine.random_actions_delay import random_actions_delay

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

#####################
#   Setup For GPU   #
#####################
device = torch.device("cpu")
#device_name = torch.cuda.get_device_name(0)
#has_gpu = torch.cuda.is_available()
#####################

#################
# Setup agents  #
#################
players[0] = DQNAgent(player_num=0, map_name=map_name, train=True)
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
final_score = 0
short_term_final_score = np.zeros((k,)) # Used to average win rates
short_term_final_scores = [0.5] # Average win rates per k episodes
q_values = 0
qVals = []
reward = {0: 0, 1: 0}
reward_decay = 0
reward_divider = 1000
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

    # Reset the reward average
    while not done:
        if i_episode % 5 == 0:
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

        #### REWARD DECAY ####
        # Setup reward decay
        won = False
        if not done:
            final_score = reward[0] # gets the final score before end of game turn
            reward[0] = 0.01 # default reward for non end of game turns
        elif reward[0] < reward[1]: # if agent loses
            reward[0] = reward[0] - reward_decay # negative reward that decays over time for losing
        else:
            if reward[0] > reward[1]:
                won = True
            reward[0] = final_score / reward_divider # positive reward for winning (scores will generally be between 300 and 3500)
        #### REWARD DECAY ####

        players[0].remember_game_state(
            prev_observation,
            observations[0],
            actions[0],
            reward[0]
        )
        
        players[0].optimize_model()
        #########################

        current_eps = players[0].epsilon


    ################################
    # End of episode agent updates #
    ################################
    players[0].end_of_episode(i_episode)

    ### Updated win calculator to reflect new reward system
    if(won):
        score += 1
        short_term_wr[(i_episode-1)%k] = 1
    else:
        losses += 1
    ###

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    scores.append(score / i_episode) ## save the most recent score
    current_wr = score / i_episode
    epsilonVals.append(current_eps)

    lossVals.append(current_loss)
    short_term_final_score[(i_episode-1)%k] = final_score
    q_values = q_values / 150
    qVals.append(q_values)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {}\tEpsilon: {:.2f}\tLR: {:.2e}\tTies: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses,current_eps, players[0].learning_rate, ties), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

        short_term_final_scores.append(np.mean(short_term_final_score))
        short_term_final_score = np.zeros((k,), dtype=int)

        #### REWARD DECAY ####
        reward_decay -= 0.5 # this reduces the reward decay after k episodes to punish the agent more for losing overtime
        #### REWARD DECAY ####

    ################################
    env.close()

#####################
# Plot final charts #
#####################
fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2)

#########################
#   Epsilon Plotting    #
#########################
par1 = ax1.twinx()
par2 = ax2.twinx()
par4 = ax2.twinx()
par3.spines["right"].set_position(("axes", 1.1))
par4.spines["right"].set_position(("axes", 1.1))
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

par3.tick_params(axis='y', colors='orange')
par4.tick_params(axis='y', colors="orange")
ax2.set_xlabel('Episode #')
#############################

#########################
#   Average Reward Plot #
#########################
ax3.plot(np.arange(0, n_episodes+1,k),short_term_final_scores)
ax3.set_ylabel('Average Final Scores')
ax3.yaxis.label.set_color('blue')
#########################

#########################
#   Average Q Val Plot  #
#########################
ax4.plot(np.arange(0, n_episodes),qVals)
ax4.set_ylabel('Average Q Values')
ax4.yaxis.label.set_color('blue')
ax4.set_xlabel('Episode #')
#########################

fig.tight_layout(pad=2.0)
plt.show()
#############################

#########