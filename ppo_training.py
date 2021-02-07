## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque 

import numpy as np

from everglades_server import server
from agents.PPO1.PPOAgent import PPOAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))
from agents.random_actions import random_actions

#from everglades-server import generate_map

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path

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
#   PPO Constants   #
#####################
N_LATENT_VAR = 128
LR = 1e-2
K_EPOCHS = 8
GAMMA = 0.99
BETAS = (0.9,0.999)
EPS_CLIP = 0.2
ACTION_DIM = 132
OBSERVATION_DIM = 105
NUM_GAMES_TILL_UPDATE = 15
UPDATE_TIMESTEP = 150 * NUM_GAMES_TILL_UPDATE
#################

#################
# Setup agents  #
#################
players[0] = PPOAgent(OBSERVATION_DIM,ACTION_DIM, N_LATENT_VAR,LR,BETAS,GAMMA,K_EPOCHS,EPS_CLIP)
names[0] = 'PPO Agent'
players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################


actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 1000
timestep = 0

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

    while not done:
        if i_episode % 5 == 0:
            env.render()

        ### Removed to save processing power
        # Print statements were taking forever
        #if debug:
        #    env.game.debug_state()
        ###

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Grab previos observation for agent
        prev_observation = observations[0]

        # Update env
        observations, reward, done, info = env.step(actions)

        timestep += 1
        #########################
        # Handle agent update   #
        #########################

        # Add in rewards and terminals 7 times to reflect the other memory additions
        # i.e. Actions are added one at a time (for a total of 7) into the memory
        for i in range(7):
            players[0].memory.rewards.append(reward[0])
            players[0].memory.is_terminals.append(done)

        # Updates agent after 150 * Number of games timesteps
        if timestep % UPDATE_TIMESTEP == 0:
            players[0].optimize_model()
            players[0].memory.clear_memory()
            timestep = 0
        #########################

        current_eps = timestep
        current_loss = players[0].loss

        #pdb.set_trace()
    #####################
    #   End Game Loop   #
    #####################

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
    lossVals.append(current_loss)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {} Epsilon: {:.2f}  Loss: {:.2f}\n'.format(i_episode,current_wr,score,losses,ties,current_eps,current_loss), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)   
    ################################
    env.close()
    #########################
    #   End Training Loop   #
    #########################


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
par1.plot(np.arange(1,n_episodes+1),lossVals,color="green",alpha=0.5)
par1.set_ylabel('Loss')
par1.yaxis.label.set_color('green')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
ax2.set_ylim([0.0,1.0])
par2.plot(np.arange(1,n_episodes+1),lossVals,color="green",alpha=0.5)
par2.set_ylabel('Loss')
par2.yaxis.label.set_color('green')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average win rate')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')
plt.show()
#############################

#########
