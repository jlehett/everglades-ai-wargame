## Static Imports
import os
import sys
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt
from collections import deque 

import numpy as np

from everglades_server import server
from DQNAgent import DQNAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))
from State_Machine.random_actions import random_actions

#from everglades-server import generate_map

#############################
# Environment Config Setup  #
#############################
map_name = "DemoMap.json"
config_dir = '../../config/'  
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
players[0] = DQNAgent(env.num_actions_per_turn, env.observation_space, 0, map_name)
names[0] = 'DQN Agent'
players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 800

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
#########################

## Training Loop
for i_episode in range(1, n_episodes+1):
    
    ## Game Loop
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
        
        ### Debug reward values
        #print("Reward: {}", reward)
        ###

        #########################
        # Handle agent update   #
        #########################
        reward[0] = players[0].set_reward(prev_observation) if players[0].set_reward(prev_observation) != 0 else reward[0]
        players[0].memory.push(prev_observation,actions[0],observations[0],reward[0])
        players[0].optimize_model()
        players[0].update_target(i_episode)
        #########################

        current_eps = players[0].eps_threshold

        #pdb.set_trace()


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
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Epsilon: {:.2f} Ties: {}\n'.format(i_episode,current_wr,score,losses,current_eps, ties), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)
        
    ################################
    env.close()

#####################
# Plot final charts #
#####################
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylim([0.0,1.0])
ax2.set_ylim([0.0,1.0])
fig.suptitle('Win rates')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative win rate')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average win rate')
ax2.set_xlabel('Episode #')
plt.show()
#####################
