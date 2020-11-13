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

from everglades_server import server
from agents.dqnagent import DQNAgent
#from everglades-server import generate_map

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
if len(sys.argv) > 2:
    agent1_file = 'agents/' + sys.argv[2]
else:
    agent1_file = 'agents/random_actions'

map_name = "DemoMap.json"
    
config_dir = 'D:\\Senior Design\\everglades-ai-wargame\\config\\'  
map_file = config_dir + map_name
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

debug = 1

## Specific Imports
agent1_name, agent1_extension = os.path.splitext(agent1_file)
agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

players[0] = DQNAgent(env.num_actions_per_turn, env.observation_space, 0, map_name)
names[0] = "DQN Agent"
players[1] = agent1_class(env.num_actions_per_turn, 1, map_name)
names[1] = agent1_class.__name__

actions = {}

## Set high episode to test convergence
# Change back to resonable setting for other testing
n_episodes= 1000

scores = []
ties = 0
losses = 0
scores_window = deque(maxlen=100) # last 100 scores
score = 0

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

        for pid in players:
            actions[pid] = players[pid].get_action( observations[pid] )
        prev_observation = observations[0]
        observations, reward, done, info = env.step(actions)
        
        ### Debug reward values
        #print("Reward: {}", reward)
        ###

        players[0].memory.push(prev_observation,actions[0],observations[0],reward[0])
        players[0].optimize_model()
        players[0].update_target(1)
        #pdb.set_trace()


    ### Updated win calculator to reflect new reward system
    if(reward[0] == 10000):
        score += 1
    elif(reward[0] == 0):
        ties += 1
    else:
        losses += 1
    ###

    scores_window.append(score / i_episode) ## save the most recent score
    scores.append(score / i_episode) ## save the most recent score
    current_wr = score / i_episode
    print('\rEpisode: {}\tCurrent WR: {:.2f}\tAverage Score: {:.2f}\tWins: {}\tLosses: {}\tTies: {}'.format(i_episode,current_wr,np.mean(scores_window),score,losses,ties), end="")
    if i_episode %100==0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
        
    if np.mean(scores_window)>=200.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        break
    env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()