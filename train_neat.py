import neat
import numpy as np
import importlib
import os
import gym
import gym_everglades
from everglades_server import server
import torch

from agents.NEAT.NEATAgent import NEATAgent
from agents.State_Machine.random_actions_delay import random_actions

MAX_GENS = 100
RENDER_TESTS = False
CHECKPOINT_GENERATION_INTERVAL = 1

neat_agent = None
n = 1


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    debug = False

    map_name = "DemoMap.json"
    config_dir = './config/'  
    map_file = config_dir + map_name
    setup_file = config_dir + 'GameSetup.json'
    unit_file = config_dir + 'UnitDefinitions.json'
    output_dir = './game_telemetry/'

    env = gym.make('everglades-v0')
    players = {}
    names = {}
    players[0] = random_actions(env.num_actions_per_turn, 1, map_name) # Dummy value for everglades
    players[1] = random_actions(env.num_actions_per_turn, 1, map_name)
    names[1] = 'Random Agent'
    names[0] = "NEAT Agent"

    global n

    for i in range(n):
        # print("--> Starting new episode")
        actions = {}

        observations = env.reset(
                players=players,
                config_dir = config_dir,
                map_file = map_file,
                unit_file = unit_file,
                output_dir = output_dir,
                pnames = names,
                debug = debug)

        actions[0] = eval_network(net, observations[0])
        actions[1] = players[1].get_action(observations[1])

        done = False
        rewards = {0: 0, 1: 0}
        final_reward = 0

        while not done:

            # env.render()
            final_reward = rewards[0]

            observations, rewards, done, info = env.step(actions)

            # print("\t Reward {}: {}".format(t, reward))

            actions[0] = eval_network(net, observations[0])
            actions[1] = players[1].get_action(observations[1])

            if done:
                # print("<-- Episode finished after {} timesteps".format(t + 1))
                # Only provide rewards after game is over
                # All fitness rewards must be non-negative for NEAT's maximization functions
                # Since rewards are based on signed percent difference, just add 1
                total_reward += final_reward + 1
                break

    return total_reward / n


def eval_network(net, net_input):
    action = np.zeros((7,2))

    # TODO: Change to work with everglades
    result = torch.from_numpy(np.asarray(net.activate(net_input)))
    result = torch.topk(result,7)[1]

    chosen_units = result // 12
    chosen_nodes = result % 11

    action[:,0] = chosen_units
    action[:,1] = chosen_nodes
    return action


def main():
    global neat_agent
    neat_agent = NEATAgent()
    neat_agent.run(eval_network,
                      eval_single_genome,
                      environment_name="everglades-v0")


if __name__ == '__main__':
    main()