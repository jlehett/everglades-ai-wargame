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
import random

import numpy as np

from everglades_server import server
from agents.RPPO.RPPOAgent import RPPOAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))
from agents.State_Machine.random_actions import random_actions

from RewardShaping import RewardShaping
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
#############################

debug = False

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

#####################
#   PPO Constants   #
#####################
N_LATENT_VAR = 248
LR = 0.0001
K_EPOCHS = 8
GAMMA = 0.99
BETAS = (0.9,0.999)
EPS_CLIP = 0.2
ACTION_DIM = 132
OBSERVATION_DIM = 105
UPDATE_TIMESTEP = 2000
LAMBD = 0.95
DEVICE = "GPU"
#################

#################
# Setup agents  #
#################
players[0] = RPPOAgent(OBSERVATION_DIM,ACTION_DIM, N_LATENT_VAR,LR,BETAS,GAMMA,K_EPOCHS,EPS_CLIP, LAMBD, DEVICE)
names[0] = 'RPPO Agent'
hidden = torch.zeros(N_LATENT_VAR).unsqueeze(0).unsqueeze(0)
players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################

#############################
#   Setup Reward Shaping    #
#############################
reward_shaper = RewardShaping()
#############################


actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 3000
timestep = 0

#########################
# Statistic variables   #
#########################
scores = []
k = 50
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates
short_term_scores = [0.5] # Average win rates per k episodes
ties = 0
losses = 0
score = 0
current_eps = 0
epsilonVals = []
current_loss = 0
lossVals = []
current_actor_loss = 0
actorLossVals = []
current_critic_loss = 0
criticLossVals = []
entropy = 0
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

    turnNum = 0
    while not done:
        if i_episode % 25 == 0:
            env.render()

        actions[1] = players[1].get_action( observations[1] )
        actions[0], hidden = players[0].get_action( observations[0], hidden )

        # Update env
        #turn_scores,_ = env.game.game_turn(actions) # Gets the score from the server
        observations, reward, done, info = env.step(actions)

        # Reward Shaping
        turn_scores = reward_shaper.reward_short_games(reward, done, turnNum)

        timestep += 1
        #########################
        # Handle agent update   #
        #########################

        # Add in rewards and terminals 7 times to reflect the other memory additions
        # i.e. Actions are added one at a time (for a total of 7) into the memory

        # Set inverse of done for is_terminals (prevents need to inverse later)
        inv_done = 1 if done == 0 else 0

        for i in range(7):
            players[0].memory.next_states.append(torch.from_numpy(observations[0]).float())
            players[0].memory.rewards.append(turn_scores)
            players[0].memory.is_terminals.append(torch.from_numpy(np.asarray(inv_done)))

        # Updates agent after UPDATE_TIMESTEP number of steps
        if timestep % UPDATE_TIMESTEP == 0:
            players[0].optimize_model()
            players[0].memory.clear_memory()
            timestep = 0

            # Reset the hidden states
            hidden = torch.zeros(N_LATENT_VAR).unsqueeze(0).unsqueeze(0)
        #########################

        current_eps = timestep
        current_loss = players[0].loss
        current_actor_loss = players[0].actor_loss
        current_critic_loss = players[0].critic_loss
        entropy = players[0].dist_entropy

        # Increment the turnNum
        turnNum += 1

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
    actorLossVals.append(current_actor_loss)
    criticLossVals.append(current_critic_loss)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.4f}\tWins: {} Losses: {} Ties: {} Steps until Update: {} Loss: {:.4f} Actor Loss: {:.4f} Critic Loss: {:.4f} Entropy: {:.4f}\n'.format(i_episode,current_wr,score,losses,ties,(UPDATE_TIMESTEP - current_eps),current_loss,current_actor_loss,current_critic_loss,entropy), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.4f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

        # Handle reward updates
        reward_shaper.update_rewards(i_episode)
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
par2 = ax2.twinx()
par3 = ax1.twinx()
par4 = ax2.twinx()
#########################

######################
#   Cumulative Plot  #
######################
fig.suptitle('Scores')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative Scores')
ax1.yaxis.label.set_color('blue')
par1.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par1.set_ylabel('Loss')
par1.yaxis.label.set_color('red')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
par2.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par2.set_ylabel('Loss')
par2.yaxis.label.set_color('red')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average Scores')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')
#############################

##################################
#   Actor Loss Plot              #
##################################
par3.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par3.set_ylabel('Loss')
par3.yaxis.label.set_color('red')
ax3.plot(np.arange(1, n_episodes+1),actorLossVals)
ax3.set_ylabel('Actor Loss')
ax3.yaxis.label.set_color('blue')
ax3.set_xlabel('Episode #')
##################################

##################################
#   Critic Loss Plot             #
##################################
par4.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par4.set_ylabel('Loss')
par4.yaxis.label.set_color('red')
ax4.plot(np.arange(1, n_episodes+1),criticLossVals)
ax4.set_ylabel('Critic Loss')
ax4.yaxis.label.set_color('blue')
ax4.set_xlabel('Episode #')
##################################

plt.show()
#########
