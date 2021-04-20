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

import utils.reward_shaping as reward_shaping
from utils.Statistics import AgentStatistics

from everglades_server import server
from agents.A2CAgent.A2CAgent import A2CAgent
from agents.A2CAgent.render_A2C import render_charts
from agents.State_Machine.random_actions import random_actions
from agents.State_Machine.random_actions_delay import random_actions_delay

#from everglades-server import generate_map

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
#if len(sys.argv) > 2:
#    agent1_file = 'agents/' + sys.argv[2]
#else:
#    agent1_file = 'agents/State_Machine/random_actions'

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

## Specific Imports
#agent1_name, agent1_extension = os.path.splitext(agent1_file)
#agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
#agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

## Constants
RENDER_CHARTS = True

#################
# Setup agents  #
#################
players[0] = A2CAgent(
    action_space=132,
    observation_space=env.observation_space,
    n_latent_var=128,
    K_epochs=4,
    gamma=0.999,
    network_save_name = '/agents/A2CAgent/saved_models/A2C_test_delayed',
    network_load_name = None
)
names[0] = "A2C Agent"
players[1] = random_actions_delay(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent Delay'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 2500

#########################
# Statistic variables   #
#########################
k = 50 # Used for average win rates
p = 5 # Print episodic results every p episodes
stats = AgentStatistics(names[0], n_episodes, k, save_file= os.getcwd() + "/saved-stats/A2C_test_delayed_stats")

scores = []
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
turn_num = 0
#########################

#####################
#   Training Loop   #
#####################
for i_episode in range(1, n_episodes+1):
    #################
    #   Game Loop   #
    #################
    turn_num = 0
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
    average_reward = 0
    while not done:
        if i_episode % 100 == 0:
            try:
                env.render()
            except:
                pass

        ### Removed to save processing power
        # Print statements were taking forever
        #if debug:
        #    env.game.debug_state()
        ###

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )


        # Update env
        observations, reward, done, info = env.step(actions)

        #########################
        # Handle agent update   #
        #########################
        #reward[0] = players[0].set_reward(prev_observation) if players[0].set_reward(prev_observation) != 0 else reward[0]
        
        # Reward short games
        if done:
            reward[0] = reward_shaping.reward_short_games(0, reward, done, turn_num)
         
        #if done:
        #    if reward[0] > reward[1]:
        #        reward[0] = (150 - turn_num) / 150
        #    else:
        #        reward[0] = -1

        

        # Update the agent's average reward
        average_reward += reward[0]

        # Unwravel action to add into memory seperately
        for i in range(7):
            players[0].memory.rewards.append(reward[0])
        
        players[0].optimize_model()

        #########################

        #current_eps = players[0].eps_threshold
        #if players[0].Temp != 0:
            #current_eps = players[0].Temp
        current_loss = players[0].loss

        turn_num = 0
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
    stats.scores.append(score / i_episode)
    current_wr = score / i_episode
    epsilonVals.append(current_eps)
    lossVals.append(current_loss)
    stats.network_loss.append(current_loss)
    average_reward /= 150 # average reward accross the 150 turns
    avgRewardVals.append(average_reward)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    if i_episode % p == 0:
        print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {} Eps/Temp: {:.2f} Loss: {:.2f} Average Reward: {:.2f}\n'.format(i_episode,current_wr,score,losses,ties,current_eps, current_loss,average_reward), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage WR {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        stats.short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)
        stats.save_stats()
  
    ################################
    try:
        env.close()
    except:
        pass
    #########################
    #   End Training Loop   #
    #########################

# Save final network
players[0].save_network(i_episode)

# Save run stats
stats.save_stats()

# Render charts to show visual of training stats
if RENDER_CHARTS:
    render_charts(stats)

'''
#####################
# Plot final charts #
#####################
fig, (ax1, ax2,ax3) = plt.subplots(3)

#########################
#   Epsilon Plotting    #
#########################
par1 = ax1.twinx()
par3 = ax1.twinx()
par2 = ax2.twinx()
par4 = ax2.twinx()
par3.spines["right"].set_position(("axes", 1.2))
par4.spines["right"].set_position(("axes", 1.2))
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
par1.set_ylabel('Eps/Temp')
par1.yaxis.label.set_color('green')
par3.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
par3.set_ylabel('Loss')
par3.yaxis.label.set_color('orange')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
ax2.set_ylim([0.0,1.0])
par2.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
par2.set_ylabel('Eps/Temp')
par2.yaxis.label.set_color('green')
par4.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
par4.set_ylabel('Loss')
par4.yaxis.label.set_color('orange')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average win rate')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')

par3.tick_params(axis='y', colors='orange')
par4.tick_params(axis='y', colors="orange")
#############################

#########################
#   Average Reward Plot #
#########################
ax3.plot(np.arange(1, n_episodes+1),avgRewardVals)
ax3.set_ylabel('Average reward')
ax3.yaxis.label.set_color('blue')
ax3.set_xlabel('Episode #')
#########################

plt.show()
#########################
#   Setup Loss Spines   #
#########################
for ax in [par3, par4]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

#########################

#########
'''