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
from agents.Rainbow2.DQNAgent import DQNAgent
from agents.Rainbow2.constants.constants import constants
from agents.State_Machine.random_actions import random_actions

constants_path = "/agents/Rainbow2/constants/constants.json"
constants = constants(constants_path)

## Main Script
env = gym.make(constants.env_name)
players = {}
names = {}

#################
# Setup agents  #
#################
players[0] = DQNAgent(player_num=0, map_name=constants.map_file)
names[0] = "DQN Agent"
players[1] = random_actions(env.num_actions_per_turn, 1, 'DemoMap.json')
names[1] = 'Random Agent Delay'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = constants.n_episodes

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
        config_dir = constants.config_dir,
        map_file = constants.map_file,
        unit_file = constants.unit_file,
        output_dir = constants.env_output_dir,
        pnames = names,
        debug = constants.debug
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
        players[0].remember_game_state(
            prev_observation,
            observations[0],
            actions[0],
            reward[0]
        )
        players[0].optimize_model()
        #########################

        #noisy net managed
        #current_eps = players[0].policy_net.advantage_layer.curr_epsilon()


    ################################
    # End of episode agent updates #
    ################################
    players[0].end_of_episode(i_episode, n_episodes)

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
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {}\n'.format(i_episode+players[0].previous_episodes,current_wr,score,losses, ties), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)
        
    ################################
    env.close()

save_file = open(constants.network_save_name + '_data.pickle', 'wb')
pickle.dump({
    'scores': scores,
    'k': k,
    'short_term_wr': short_term_wr,
    'short_term_scores': short_term_scores,
    'ties': ties,
    'losses': losses,
    'score': score,
    'current_eps': current_eps,
    'epsilonVals': epsilonVals,
    'current_loss': current_loss,
    'lossVals': lossVals,
    'average_reward': average_reward,
    'avgRewardVals': avgRewardVals,
}, save_file)
save_file.close()

# #####################
# # Plot final charts #
# #####################
# fig, (ax1, ax2) = plt.subplots(2)

# #########################
# #   Epsilon Plotting    #
# #########################
# par1 = ax1.twinx()
# par2 = ax2.twinx()
# #########################

# ######################
# #   Cumulative Plot  #
# ######################
# ax1.set_ylim([0.0,1.0])
# fig.suptitle('Win rates')
# ax1.plot(np.arange(1, n_episodes+1),scores)
# ax1.set_ylabel('Cumulative win rate')
# ax1.yaxis.label.set_color('blue')
# par1.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
# par1.set_ylabel('Epsilon')
# par1.yaxis.label.set_color('green')
# #######################

# ##################################
# #   Average Per K Episodes Plot  #
# ##################################
# ax2.set_ylim([0.0,1.0])
# par2.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
# par2.set_ylabel('Epsilon')
# par2.yaxis.label.set_color('green')
# ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
# ax2.set_ylabel('Average win rate')
# ax2.yaxis.label.set_color('blue')
# ax2.set_xlabel('Episode #')
# plt.show()
# #############################

# #########