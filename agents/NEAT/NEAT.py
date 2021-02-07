import gym
import math
import os
import importlib
import gym_everglades
from everglades_server import server
import pdb
import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from collections import namedtuple, deque
from itertools import count
import torch

from pathlib import Path

import neat
import pickle

class NEATTrainer():
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
    
    #####################################
    #   Worker function to train NEAT   #
    #####################################
    def run(self):
        ## Input Variables
        # Agent files must include a class of the same name with a 'get_action' function
        # Do not include './' in file path
        agent1_file = 'random_actions'

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
        agent1_name, agent1_extension = os.path.splitext(agent1_file)
        agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
        agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

        ## Main Script
        env = gym.make('everglades-v0')
        players = {}
        names = {}

        ########################
        # Setup Random agents  #
        ########################
        players[1] = agent1_class(env.num_actions_per_turn, 1, map_name)
        names[1] = agent1_class.__name__
        #################

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
        #########################


        actions = {}

        ## Set high episode to test convergence
        # Change back to resonable setting for other testing
        n_episodes = 800

        #####################
        #   Training Loop   #
        #####################
        for genome_id, genome in self.genome:
            #########################
            #   Setup NEAT Agent    #
            #########################
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            players[0] = net
            names[0] = "NEAT Agent"
            #########################


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
                #if i_episode % 5 == 0:
                #    env.render()

                ### Removed to save processing power
                # Print statements were taking forever
                #if debug:
                #    env.game.debug_state()
                ###

                # Get actions for each player
                # NEAT Action
                neat_action = players[0].activate(observations[0])
                neat_action = torch.FloatTensor(neat_action)
                neat_action = torch.reshape(neat_action,(12,11))
                actions[0] = NEATTrainer.unwravelActions(neat_action)
                # Other Agent Action
                actions[1] = players[1].get_action(observations[1])

                # Update env
                observations, reward, done, info = env.step(actions)

                if(not done):
                    prev_reward = reward[0]

                #########################
                # Handle agent update   #
                #########################

                #########################

                #pdb.set_trace()
            #####################
            #   End Game Loop   #
            #####################

            ### Updated win calculator to reflect new reward system
            if(reward[0] > reward[1]):
                score = 1
                if(genome.fitness != None):
                    genome.fitness += 1
                else:
                    genome.fitness = 1
                #short_term_wr[(i_episode-1)%k] = 1
            elif(reward[0] == reward[1]):
                ties += 1
            else:
                losses += 1
                if(genome.fitness != None):
                    genome.fitness -= 1
                else:
                    genome.fitness = -1
            ###

            #####################################
            #   Set population members fitness  #
            #####################################
            genome.fitness = reward[0] if (reward[0] > prev_reward) else prev_reward
            #####################################

            #############################################
            # Update Score statistics for final chart   #
            #############################################
            #scores.append(score / i_episode) ## save the most recent score
            #current_wr = score / i_episode
            #############################################

            #################################
            # Print current run statistics  #
            #################################
            print('Genome_ID: {}\tFitness: {}\n'.format(genome_id,genome.fitness))
            #print('\rEpisode: {}\tCurrent WR: {:.2f}\tWins: {}\tLosses: {} Ties: {}\n'.format(i_episode,current_wr,score,losses, ties), end="")
            #if i_episode % k == 0:
            #    print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(short_term_wr)))
            #    short_term_scores.append(np.mean(short_term_wr))
            #    short_term_wr = np.zeros((k,), dtype=int)   
            ################################
            env.close()
            #########################
            #   End Training Loop   #
            #########################
    
    def unwravelActions(actionTensor):
        # Initialize unit, node and q-value arrays
        best_action_units = np.zeros(7)
        best_action_nodes = np.zeros(7)
        best_action_qs = np.zeros(7)
        action = np.zeros((7,2))
        # Unravel the output tensor into two size 7 arrays
        for group_index in range(12):
            for node_index in range(11):
                for action_index in range(7):
                    # Get largest q-value actions
                    # Discard if lower than another action
                    if actionTensor[group_index, node_index] > best_action_qs[action_index]:
                        # Prevent unit numbers from appearing in best_action_units multiple times
                        if group_index in best_action_units and best_action_units[action_index] != group_index:
                            continue
                        else:
                            best_action_qs[action_index] = actionTensor[group_index, node_index]
                            best_action_units[action_index] = group_index
                            best_action_nodes[action_index] = node_index
                            break
        
        # Create the final action array to return in a readable format
        action[:, 0] = best_action_units
        action[:, 1] = best_action_nodes
        return action

def eval_genomes(genome, config):
    
    train = NEATTrainer(genome, config)
    return train.run()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     './agents/NEAT/neat-config.txt')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

#pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(eval_genomes,100)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)