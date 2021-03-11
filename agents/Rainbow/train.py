## Static Imports
import os
import importlib
import gym
import gym_everglades
from everglades_server import server
import pdb
import sys
from collections import deque
import numpy as np
from data.graph import graph
from constants.constants import constants
from agent import DQNAgent
from State_Machine.random_actions_delay import random_actions_delay

constants_path = "/constants/constants.json"
constants = constants(constants_path)

debug = constants.debug

## Main Script
env = gym.make(constants.env_name)
players = {}
names = {}

#################
# Setup agents  #
#################
players[0] = DQNAgent(0, constants.map_file, env.observation_space)
names[0] = "DQN Agent"
players[1] = random_actions_delay(env.num_actions_per_turn, 1, constants.map_file)
names[1] = 'Random Agent'
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
final_score = 0
short_term_final_score = np.zeros((k,)) # Used to average win rates
short_term_final_scores = [0.5] # Average win rates per k episodes
q_values = 0
qVals = []
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
        config_dir = constants.config_dir,
        map_file = constants.map_file,
        unit_file = constants.unit_file,
        output_dir = constants.env_output_dir,
        pnames = names,
        debug = debug
    )

    while not done:
        if i_episode % 25 == 0:
            env.render()

        ### Removed to save processing power
        # Print statements were taking forever
        #if debug:
        #    env.game.debug_state()
        ###

        # Get agents final score before end of game reward
        final_score = reward[0]

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
        reward[0] = players[0].set_reward(prev_observation) if players[0].set_reward(prev_observation) != 0 else reward[0]

        # Unwravel action to add into memory seperately
        action_0 = 0
        for i in range(7):
            action_0 = (actions[0][i][0] * 11 + actions[0][i][1]) 
            players[0].memory.push(prev_observation,action_0,observations[0],reward[0])
        
        players[0].optimize_model()
        players[0].update_target(i_episode)
        #########################

        current_eps = players[0].eps_threshold
        if players[0].Temp != 0:
            current_eps = players[0].Temp
        q_values += players[0].q_values.mean()
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
    short_term_final_score[(i_episode-1)%k] = final_score
    q_values = q_values / 150
    qVals.append(q_values)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {} Eps/Temp: {:.2f} Loss: {:.2f} Average Q-Value: {:.2f} Final Score: {:.2f}\n'.format(i_episode,current_wr,score,losses,ties,current_eps, current_loss,q_values,final_score), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)
        short_term_final_scores.append(np.mean(short_term_final_score))    
        short_term_final_score = np.zeros((k,))
    ################################
    env.close()
    #########################
    #   End Training Loop   #
    #########################

graph(n_episodes, scores, epsilonVals, lossVals, k, short_term_scores, short_term_final_scores, qVals)