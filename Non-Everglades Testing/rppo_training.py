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
from RPPO import RPPOAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))

#from RewardShaping import RewardShaping

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
env = gym.make('CartPole-v1')
players = {}
names = {}

#####################
#   PPO Constants   #
#####################
N_LATENT_VAR = 64
LR = 0.0001
K_EPOCHS = 4
GAMMA = 0.99
BETAS = (0.9,0.999)
EPS_CLIP = 0.2
ACTION_DIM = env.action_space.n
OBSERVATION_DIM = env.observation_space.shape[0] - 1# Removing one observation feature to make POMDP
NUM_GAMES_TILL_UPDATE = 25
UPDATE_TIMESTEP = 2000#NUM_GAMES_TILL_UPDATE
INTR_REWARD_STRENGTH = 0.9
ICM_BATCH_SIZE = 200
TARGET_KL = 0.01
LAMBD = 0.95
USE_ICM = False
USE_GRU = True
#################

#################
# Setup agents  #
#################
players[0] = RPPOAgent(OBSERVATION_DIM,ACTION_DIM, N_LATENT_VAR,LR,BETAS,GAMMA,K_EPOCHS,EPS_CLIP, 
                        INTR_REWARD_STRENGTH, ICM_BATCH_SIZE, TARGET_KL, LAMBD, USE_ICM, USE_GRU)
names[0] = 'RPPO Agent'
hidden = torch.zeros(N_LATENT_VAR).unsqueeze(0).unsqueeze(0)
#################

#############################
#   Setup Reward Shaping    #
#############################
#reward_shaper = RewardShaping()
#############################


actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 5000
loss_reward_decay = {0: -1} #reward decay for losing
timestep = 0
episodes_to_update = 0

#########################
# Statistic variables   #
#########################
scores = []
k = 50#NUM_GAMES_TILL_UPDATE
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
current_temp = 0
tempVals = []
just_updated = True
#########################

#####################
#   Training Loop   #
#####################
for i_episode in range(1, n_episodes+1):
    #################
    #   Game Loop   #
    #################

    done = 0
    observations = env.reset()
    # Remove cart velocity to turn into POMDP
    observations = np.delete(observations,1,0)
    #observations = np.delete(observations,1,0)

    #players[1].reset()
    score = 0
    cumulative_reward = 0
    while not done:
        #if i_episode % 25 == 0:
        #    env.render()

        ### Removed to save processing power
        # Print statements were taking forever
        #if debug:
        #    env.game.debug_state()
        ###

        action, hidden = players[0].get_action( observations, hidden )

        # Update env
        observations, reward, done, info = env.step(action)

        if not done:
            reward = 0.01
            #hidden = hidden.detach()
        else:
            # Iterate over loss_reward_decay to find nearest (but larger) required reward (key)
            # Use that value as the reward decay
            # If none found, give default decay
            #found_less = False
            #for required_reward in loss_reward_decay.keys():
            #    if score < required_reward:
            #        reward = loss_reward_decay.get(required_reward)
            #        found_less = True
            #        break
            
            #if not found_less:
            reward = -1

        # Remove cart velocity to turn into POMDP
        observations = np.delete(observations,1,0)
        #observations = np.delete(observations,1,0)

        # Reward Shaping
        #won,reward,final_score,final_score_random = reward_shaper.get_reward(done, reward)

        timestep += 1
        cumulative_reward += reward
        #########################
        # Handle agent update   #
        #########################

        # Add in rewards and terminals 7 times to reflect the other memory additions
        # i.e. Actions are added one at a time (for a total of 7) into the memory

        # Set inverse of done for is_terminals (prevents need to inverse later)
        inv_done = 1 if done == 0 else 0

        players[0].memory.next_states.append(torch.from_numpy(observations).float())
        players[0].memory.rewards.append(reward)
        players[0].memory.is_terminals.append(torch.from_numpy(np.asarray(inv_done)))

        if timestep % UPDATE_TIMESTEP == 0:
            players[0].optimize_model()
            players[0].memory.clear_memory()
            timestep = 0
            episodes_to_update = 0
            # Reset the hidden states using an updated
            hidden = torch.zeros(N_LATENT_VAR).unsqueeze(0).unsqueeze(0)

        #########################

        current_eps = timestep
        current_loss = players[0].loss
        current_actor_loss = players[0].actor_loss
        current_critic_loss = players[0].critic_loss
        current_temp = players[0].temperature

        if not done:
            score += 1

        #pdb.set_trace()
    #####################
    #   End Game Loop   #
    #####################
    episodes_to_update += 1

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    scores.append(score) ## save the most recent score
    short_term_wr[(i_episode-1)%k] = score
    current_wr = score / i_episode
    epsilonVals.append(current_eps)
    lossVals.append(current_loss)
    actorLossVals.append(current_actor_loss)
    criticLossVals.append(current_critic_loss)
    tempVals.append(current_temp)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tScore: {:.2f} Final Reward: {:.2f} Epsilon: {:.2f} Temperature: {:.2f}  Loss: {:.4f} Actor Loss: {:.4f} Critic Loss: {:.4f} Inv_Loss: {:.2f} Fwd_Loss: {:.2f}\n'.format(i_episode,score,cumulative_reward,current_eps,current_temp,current_loss,current_actor_loss,current_critic_loss,players[0].inv_loss, players[0].forward_loss), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

        # Handle reward updates
        #reward_shaper.update_rewards(i_episode)

        # Update reward decay values
        # First iterate and decrease all decay values by 0.5
        new_requirement = 0
        new_decay = -100
        for required_reward in loss_reward_decay.keys():
            if loss_reward_decay.get(required_reward) > new_decay:
                new_decay = loss_reward_decay.get(required_reward)

            loss_reward_decay[required_reward] -= 0.5

            if required_reward > new_requirement:
                new_requirement = required_reward
        # Add new loss_reward_decay for next final_requirement
        new_requirement += 50
        loss_reward_decay[new_requirement] = -0.5
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
par5 = ax3.twinx()
par6 = ax4.twinx()
par3.spines["right"].set_position(("axes", 1.1))
par4.spines["right"].set_position(("axes", 1.1))
#########################

######################
#   Cumulative Plot  #
######################
fig.suptitle('Scores')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative Scores')
ax1.yaxis.label.set_color('blue')
par1.plot(np.arange(1,n_episodes+1),tempVals,color="green",alpha=0.5)
par1.set_ylabel('Temperature')
par1.yaxis.label.set_color('green')
par3.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par3.set_ylabel('Loss')
par3.yaxis.label.set_color('red')
#######################

##################################
#   Average Per K Episodes Plot  #
##################################
par2.plot(np.arange(1,n_episodes+1),tempVals,color="green",alpha=0.5)
par2.set_ylabel('Temperature')
par2.yaxis.label.set_color('green')
par4.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par4.set_ylabel('Loss')
par4.yaxis.label.set_color('red')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average Scores')
ax2.yaxis.label.set_color('blue')
ax2.set_xlabel('Episode #')
#############################

##################################
#   Actor Loss Plot              #
##################################
par5.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par5.set_ylabel('Loss')
par5.yaxis.label.set_color('red')
ax3.plot(np.arange(1, n_episodes+1),actorLossVals)
ax3.set_ylabel('Actor Loss')
ax3.yaxis.label.set_color('blue')
ax3.set_xlabel('Episode #')
##################################

##################################
#   Critic Loss Plot             #
##################################
par6.plot(np.arange(1,n_episodes+1),lossVals,color="red",alpha=0.5)
par6.set_ylabel('Loss')
par6.yaxis.label.set_color('red')
ax4.plot(np.arange(1, n_episodes+1),criticLossVals)
ax4.set_ylabel('Critic Loss')
ax4.yaxis.label.set_color('blue')
ax4.set_xlabel('Episode #')
##################################

plt.show()
#########
