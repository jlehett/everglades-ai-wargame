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
from agents.DQN.DQNAgent import DQNAgent
from agents.State_Machine.random_actions import random_actions

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

#########################
#   Setup DQN Constants #
#########################
LR = 1e-6
REPLAY_SIZE = 100000
BATCH_SIZE = 256 # Updated
GAMMA = 0.99
LEAKY_SLOPE = 0.01 # Updated
WEIGHT_DECAY = 0 # Leave at zero. Weight decay has so far caused massive collapse in network output
EXPLORATION = "EPS" # Defines the exploration type. EPS is Epsilon Greedy, Boltzmann is Boltzmann Distribution Sampling
### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 0.00005
###
TARGET_UPDATE = 100 # Updated
#########################

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
players[0] = DQNAgent(env.num_actions_per_turn, env.observation_space,0,LR,REPLAY_SIZE,BATCH_SIZE,
                        GAMMA,WEIGHT_DECAY,EXPLORATION,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE)
names[0] = "DQN Agent"
players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################


actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 1000

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

    players[1].reset()

    while not done:
        if i_episode % 25 == 0:
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

        #########################
        # Handle agent update   #
        #########################

        #### REWARD DECAY ####
        # Setup reward decay
        if not done:
            final_score = reward[0] # gets the final score before end of game turn
            reward[0] = 0.01 # default reward for non end of game turns
        elif reward[0] < reward[1]: # if agent loses
            reward[0] = reward[0] - reward_decay # negative reward that decays over time for losing
        else:
            reward[0] = final_score / reward_divider # positive reward for winning (scores will generally be between 300 and 3500)
        #### REWARD DECAY ####

        reward_0 = torch.Tensor(np.asarray(reward[0]))

        if not reward_0.dim() > 0: # Puts reward as 1 dim tensor
            reward_0 = reward_0.unsqueeze(0)
        else: # Fixes error where empty tensor is passed to memory
            reward_0 = torch.ones(1) * 0
        reward_0 = reward_0.to(device) # Send to device

        batch_actions = np.zeros(7)

        # Unwravel actions
        action_0 = 0
        for i in range(7):
            batch_actions[i] = (actions[0][i][0] * 11 + actions[0][i][1])

        batch_actions = torch.from_numpy(batch_actions).to(device) # Send batched actions to gpu
        players[0].memory.push(torch.from_numpy(prev_observation).to(device),batch_actions, # Add all to memory
            torch.from_numpy(observations[0]).to(device),reward_0)
        
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
        short_term_final_score = np.zeros((k,), dtype=int)

        #### REWARD DECAY ####
        reward_decay -= 0.5 # this reduces the reward decay after k episodes to punish the agent more for losing overtime
        #### REWARD DECAY ####

    ################################
    env.close()
    #########################
    #   End Training Loop   #
    #########################


#####################
# Plot final charts #
#####################
fig, ((ax1, ax3),(ax2,ax4)) = plt.subplots(2,2)

#########################
#   Epsilon Plotting    #
#########################
par1 = ax1.twinx()
par3 = ax1.twinx()
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
