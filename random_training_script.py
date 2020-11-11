## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys
import matplotlib.pyplot as plt

import numpy as np

from everglades_server import server

def plotAgentPerformance(win_rate, num_games, save=False):
    plt.figure()
    plt.plot(
        [0, num_games+1],
        [50, 50],
        'r--'
    )
    plt.plot(
        [episode for episode in range(1, len(win_rate)+1)],
        agent_0_win_rates
    )
    plt.title('DQN Agent Win Rate Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (%)')
    plt.xlim(left=1, right=num_games)
    plt.yticks([i*10 for i in range(11)])
    if save:
        plt.savefig('performance.png', orientation='landscape')
    else:
        plt.show()

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
training_agent_file = 'agents/' + sys.argv[1]
random_agent_file = 'agents/random_actions_delay'

map_name = "DemoMap.json"
    
config_dir = './config/'  
map_file = config_dir + map_name
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

## Specific Imports
agent0_name, agent0_extension = os.path.splitext(training_agent_file)
agent0_mod = importlib.import_module(agent0_name.replace('/','.'))
agent0_class = getattr(agent0_mod, os.path.basename(agent0_name))

agent1_name, agent1_extension = os.path.splitext(random_agent_file)
agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

players[0] = agent0_class(
    env=env,
    map_name=map_name,
    h=100,
    lr=10.0,
    epsilon=0.0,
    epsilon_decay=0.95,
    discount=0.0,
)
names[0] = agent0_class.__name__
players[1] = agent1_class(env.num_actions_per_turn, 1, map_name)
names[1] = agent1_class.__name__

# Training Params
num_games = 10000
display_game_after = 50

agent_0_wins = 0
agent_0_win_rates = []
# Run training for the num_games
for game_num in range(1, num_games+1):
    # Reset game state
    previous_observations = env.reset(
        players=players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = None,
        pnames = names,
        debug = False
    )

    actions = {}
    # Game Loop
    done = 0
    while not done:
        if display_game_after and game_num >= display_game_after:
            env.render()
        
        for pid in players:
            actions[pid] = players[pid].get_action( previous_observations[pid] )

        next_observations, reward, done, info = env.step(actions)

        # Train the agent using the previous observations, the action the agent took, and the
        # reward it received for taking the action given the previous observations.
        if done:
            players[0].train(
                previous_state=previous_observations[0],
                actions=actions[0],
                reward=reward[0],
            )
        else:
            players[0].train(
                previous_state=previous_observations[0],
                next_state=next_observations[0],
                actions=actions[0],
                reward=reward[0],
            )

        # Reset the previous states
        previous_observations = next_observations

    # Close the environment rendering window
    env.close()

    # Handle end-of-episode logic
    players[0].endOfEpisode()
    if reward[0] == 1:
        agent_0_wins += 1
    agent_0_wr = agent_0_wins / game_num * 100.0
    print('Game ' + str(game_num) + ' / ' + str(num_games) + ' played.')
    print('Win Rate: ' + str(agent_0_wr))
    print('\n')
    agent_0_win_rates.append(agent_0_wr)

    if game_num % 50 == 0:
        plotAgentPerformance(agent_0_win_rates, num_games, save=True)