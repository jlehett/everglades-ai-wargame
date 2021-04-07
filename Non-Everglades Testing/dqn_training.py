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

from DQNAgent import DQNAgent

debug = False

## Main Script
env = gym.make('CartPole-v1')
player = None
name = None

#########################
#   Setup DQN Constants #
#########################
LR = 0.00001
REPLAY_SIZE = 1000
BATCH_SIZE = 32 # Updated
GAMMA = 0.99
LEAKY_SLOPE = 0.01 # Updated
WEIGHT_DECAY = 0 # Leave at zero. Weight decay has so far caused massive collapse in network output
EXPLORATION = "EPS" # Defines the exploration type. EPS is Epsilon Greedy, Boltzmann is Boltzmann Distribution Sampling
### Updated the decay to finish later
# Increase by one order of magnitude to finish around episode 200-250
EPS_START = 0.5
EPS_END = 0.01
EPS_DECAY = 0.0001
###
TARGET_UPDATE = 10 # Updated
#########################

#####################
#   Setup For GPU   #
#####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0)
has_gpu = torch.cuda.is_available()
#####################

#################
# Setup agents  #
#################
player = DQNAgent(env.action_space, env.observation_space,0,LR,REPLAY_SIZE,BATCH_SIZE,
                        GAMMA,WEIGHT_DECAY,EXPLORATION,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE)
name = "DQN Agent"
#################


action = None

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 1000

loss_reward_decay = {0: -1} #reward decay for losing

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
final_reward = 0
final_rewards = []
q_values = 0
qVals = []
reward = 0
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
    score = 0
    cumulative_reward = 0
    while not done:
        if i_episode % 25 == 0:
            env.render()

        ### Removed to save processing power
        # Print statements were taking forever
        #if debug:
        #    env.game.debug_state()
        ###

        # Get actions for each player
        action = player.get_action( observations )

        # Grab previos observation for agent
        prev_observation = observations

        # Update env
        observations, reward, done, info = env.step(action)

        if not done:
            reward = 0.01
        else:
            # Iterate over loss_reward_decay to find nearest (but larger) required reward (key)
            # Use that value as the reward decay
            # If none found, give default decay
            found_less = False
            for required_reward in loss_reward_decay.keys():
                if score < required_reward:
                    reward = loss_reward_decay.get(required_reward)
                    found_less = True
                    break
            
            if not found_less:
                reward = -1

        cumulative_reward += reward
        #########################
        # Handle agent update   #
        #########################
        reward = np.asarray(reward)
        reward_0 = torch.from_numpy(reward).to(device)

        action_tensor = torch.tensor(action).to(device) # Send batched actions to gpu
        player.memory.push(torch.from_numpy(prev_observation).to(device),action_tensor, # Add all to memory
            torch.from_numpy(observations).to(device),reward_0)
        
        player.optimize_model()
        player.update_target(i_episode)
        #########################

        current_eps = player.eps_threshold
        q_values += player.q_values.mean()
        current_loss = player.loss

        if not done:
            score += 1

        #pdb.set_trace()
    #####################
    #   End Game Loop   #
    #####################

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    scores.append(score) ## save the most recent score
    short_term_wr[(i_episode-1)%k] = score
    final_reward = reward
    final_rewards.append(final_reward)
    epsilonVals.append(current_eps)
    lossVals.append(current_loss)
    q_values = np.mean(q_values)
    qVals.append(q_values)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    print('\rEpisode: {}\tScore: {:.2f}\tFinal Reward {:.2f} Eps/Temp: {:.2f} Loss: {:.2f} Average Q-Value: {:.2f}\n'.format(i_episode,score,cumulative_reward,current_eps, current_loss,q_values), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
        short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

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
par3 = ax1.twinx()
par2 = ax2.twinx()
par4 = ax2.twinx()
par3.spines["right"].set_position(("axes", 1.1))
par4.spines["right"].set_position(("axes", 1.1))
#########################

######################
#   Cumulative Plot  #
######################
fig.suptitle('Score')
ax1.plot(np.arange(1, n_episodes+1),scores)
ax1.set_ylabel('Cumulative Score')
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
par2.plot(np.arange(1,n_episodes+1),epsilonVals,color="green")
par2.set_ylabel('Eps/Temp')
par2.yaxis.label.set_color('green')
par4.plot(np.arange(1,n_episodes+1),lossVals,color="orange",alpha=0.5)
par4.set_ylabel('Loss')
par4.yaxis.label.set_color('orange')
ax2.plot(np.arange(0, n_episodes+1, k),short_term_scores)
ax2.set_ylabel('Average Score')
ax2.yaxis.label.set_color('blue')

par3.tick_params(axis='y', colors='orange')
par4.tick_params(axis='y', colors="orange")
ax2.set_xlabel('Episode #')
#############################

#########################
#   Average Reward Plot #
#########################
ax3.plot(np.arange(1, n_episodes+1),final_rewards)
ax3.set_ylabel('Final Reward')
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
