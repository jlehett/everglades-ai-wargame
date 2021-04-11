## Static Imports
import os, sys
sys.path.insert(0, '.')

import gym
import gym_everglades
import sys
import torch
import random
import numpy as np

# Import everglades
from everglades_server import server

# Import the agent
from agents.PPO.PPOAgent import PPOAgent

# Import agent to train against
sys.path.append(os.path.abspath('../'))
from agents.State_Machine.random_actions import random_actions

# Import utilities
import utils.reward_shaping as reward_shaping
from utils.Statistics import AgentStatistics
# from agents.PPO.render_ppo import render_charts

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
NETWORK_SAVE_NAME = "/agents/PPO/saved_models/rppo_newton_v1"
SAVE_AFTER_EPISODE = 100
USE_RECURRENT = True
TRAIN = True
DEVICE = "GPU"
#################

#################
# Setup agents  #
#################
players[0] = PPOAgent(OBSERVATION_DIM,
                ACTION_DIM, 
                N_LATENT_VAR,
                LR,
                BETAS,
                GAMMA,
                UPDATE_TIMESTEP,
                K_EPOCHS,
                EPS_CLIP, 
                LAMBD,
                USE_RECURRENT, 
                DEVICE, 
                TRAIN,
                SAVE_AFTER_EPISODE,
                NETWORK_SAVE_NAME)
names[0] = 'R/PPO Agent'
players[1] = random_actions_delay(env.num_actions_per_turn, 1, map_name)
names[1] = 'Random Agent'
#################

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes = 100000
RENDER_CHARTS = False # Determines if final charts should be rendered
timestep = 0

#########################
# Statistic variables   #
#########################
k = 100 #The set number of episodes to show win rates for
p = 10000
# The Stats class (for saving statistics)
stats = AgentStatistics(names[0], n_episodes, k, save_file="/saved-stats/rppo_newton_v1")

# General stats
score = 0
losses = 0
ties = 0

# Short wr
short_term_wr = np.zeros((k,), dtype=int) # Used to average win rates

# Epsilon and losses
current_eps = 0
current_loss = 0
current_actor_loss = 0
current_critic_loss = 0
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
        #if i_episode % 25 == 0:
        #    env.render()

        # Get actions for each player
        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )

        # Update env
        observations, reward, done, info = env.step(actions)

        # Reward Shaping
        turn_scores = reward_shaping.reward_short_games(1,reward, done, turnNum)

        timestep += 1
        #########################
        # Handle agent update   #
        #########################
        players[0].remember_game_state(observations[0], turn_scores, done)

        # Handle end of game updates
        if done:
            players[0].end_of_episode(i_episode)

        # Updates agent after UPDATE_TIMESTEP number of steps
        if timestep % UPDATE_TIMESTEP == 0:
            players[0].optimize_model()
            players[0].memory.clear_memory()
            timestep = 0
        #########################

        current_eps = timestep
        current_loss = players[0].loss
        current_actor_loss = players[0].actor_loss
        current_critic_loss = players[0].critic_loss
        entropy = players[0].dist_entropy

        # Increment the turnNum
        turnNum += 1
    #####################
    #   End Game Loop   #
    #####################

    ### Updated win calculator to reflect new reward system
    if(reward[0] > reward[1]):
        score += 1
        stats.wins += 1
        short_term_wr[(i_episode-1)%k] = 1
    elif(reward[0] == reward[1]):
        ties += 1
        stats.ties += 1
    else:
        losses += 1
        stats.losses += 1
    ###

    #############################################
    # Update Score statistics for final chart   #
    #############################################
    stats.scores.append(score / i_episode) ## save the most recent score
    current_wr = score / i_episode
    stats.epsilons.append(current_eps)
    stats.network_loss.append(current_loss)
    stats.actor_loss.append(current_actor_loss)
    stats.critic_loss.append(current_critic_loss)
    #############################################

    #################################
    # Print current run statistics  #
    #################################
    if i_episode % p == 0:
        print('\rEpisode: {}\tCurrent WR: {:.4f}\tWins: {} Losses: {} Ties: {} Steps until Update: {} Loss: {:.4f} Actor Loss: {:.4f} Critic Loss: {:.4f} Entropy: {:.4f}\n'.format(i_episode,current_wr,score,losses,ties,(UPDATE_TIMESTEP - current_eps),current_loss,current_actor_loss,current_critic_loss,entropy), end="")
    if i_episode % k == 0:
        print('\rEpisode {}\tAverage Score {:.4f}'.format(i_episode,np.mean(short_term_wr)))
        stats.short_term_scores.append(np.mean(short_term_wr))
        short_term_wr = np.zeros((k,), dtype=int)

        # Save statistics every k episodes
        stats.save_stats()
    ################################
    env.close()

    #########################
    #   End Training Loop   #
    #########################

#####################
#   FINAL STEPS     #
#####################

# Save final model state
players[0].save_network(i_episode)

# Save run stats
stats.save_stats()

# Render charts to show visual of training stats
if RENDER_CHARTS:
    render_charts(stats)