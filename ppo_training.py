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
from agents.PPO1.PPOAgent import PPOAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))
from agents.State_Machine.random_actions_delay import random_actions_delay

from RewardShaping import RewardShaping

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
N_LATENT_VAR = 128 # Number of nodes in hidden layers
LR = 0.001 
K_EPOCHS = 4 # Number of PPO epochs
GAMMA = 0.99 # For Advantage estimation
BETAS = (0.9,0.999) # For Adam
EPS_CLIP = 0.2 # PPO EPS Value
ACTION_DIM = 132
OBSERVATION_DIM = 105
NUM_GAMES_TILL_UPDATE = 10 # NOT CURRENTLY BEING UTILIZED, MAY USE LATER
UPDATE_TIMESTEP = 2000 # Time until PPO update
INTR_REWARD_STRENGTH = 0.8 # For ICM reward strength
ICM_BATCH_SIZE = 500 # For ICM Update
TARGET_KL = 0.01 # For KL Divergence Early Stopping
LAMBD = 0.95 # For Generalized Advantage Estimation
USE_ICM = False # True uses ICM
BULK_UPDATE = False # True uses bulk update (updating with the 7 actions put together in memory as opposed to separately)
#################

#################
# Setup agents  #
#################
players[0] = PPOAgent(OBSERVATION_DIM,ACTION_DIM, N_LATENT_VAR,LR,BETAS,GAMMA,K_EPOCHS,EPS_CLIP,
                        INTR_REWARD_STRENGTH, ICM_BATCH_SIZE, TARGET_KL, LAMBD, USE_ICM, BULK_UPDATE)
names[0] = 'PPO Agent'
players[1] = random_actions_delay(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################

#############################
#   Setup Reward Shaping    #
#############################
reward_shaper = RewardShaping()
#############################


actions = {}

n_episodes = 1000
timestep = 0

#########################
# Statistic variables   #
#########################
scores = []
k = 50 # Only has an affect on the average win rate graph
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates
short_term_scores = [0.5] # Average win rates per k episodes
ties = 0
losses = 0
score = 0
current_eps = 0
epsilonVals = []
current_loss = 0
lossVals = []
current_temp = 0
tempVals = []
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
        if i_episode % 25 == 0:
            env.render()

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Update env
        turn_scores,_ = env.game.game_turn(actions) # Gets the score from the server
        observations, reward, done, info = env.step(actions)

        # Reward Shaping
        won,turn_scores,final_score,final_score_random = reward_shaper.get_reward(done, turn_scores)

        timestep += 1
        #########################
        # Handle agent update   #
        #########################

        # Add in rewards and terminals 7 times to reflect the other memory additions
        # i.e. Actions are added one at a time (for a total of 7) into the memory

        # Set inverse of done for is_terminals (prevents need to inverse later)
        inv_done = 1 if done == 0 else 0

        if not BULK_UPDATE:
            for i in range(7):
                players[0].memory.next_states.append(torch.from_numpy(observations[0]).float())
                players[0].memory.rewards.append(turn_scores[0])
                players[0].memory.is_terminals.append(torch.from_numpy(np.asarray(inv_done)))
        else:
            players[0].memory.next_states.append(torch.from_numpy(observations[0]).float())
            players[0].memory.rewards.append(turn_scores[0])
            players[0].memory.is_terminals.append(torch.from_numpy(np.asarray(inv_done)))

        # Updates agent after 150 * Number of games timesteps
        if timestep % UPDATE_TIMESTEP == 0:
            players[0].optimize_model()
            players[0].memory.clear_memory()
            timestep = 0
        #########################

        current_eps = timestep
        current_loss = players[0].loss
        current_temp = players[0].temperature

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
    tempVals.append(current_temp)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {} Epsilon: {:.2f} Temperature: {:.2f}  Loss: {:.2f} Inv_Loss: {:.2f} Fwd_Loss: {:.2f}\n'.format(i_episode,current_wr,score,losses,ties,current_eps,current_temp,current_loss,players[0].inv_loss, players[0].forward_loss), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
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
fig, (ax1, ax2) = plt.subplots(2)

#########################
#   Epsilon Plotting    #
#########################
par1 = ax1.twinx()
par2 = ax2.twinx()
par3 = ax1.twinx()
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
par1.plot(np.arange(1,n_episodes+1),tempVals,color="green",alpha=0.5)
par1.set_ylabel('Temperature')
par1.yaxis.label.set_color('green')
par3.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
par3.set_ylabel('Loss')
par3.yaxis.label.set_color('orange')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
ax2.set_ylim([0.0,1.0])
par2.plot(np.arange(1,n_episodes+1),tempVals,color="green",alpha=0.5)
par2.set_ylabel('Temperature')
par2.yaxis.label.set_color('green')
par4.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
par4.set_ylabel('Loss')
par4.yaxis.label.set_color('orange')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average win rate')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')
plt.show()
#############################

#########
