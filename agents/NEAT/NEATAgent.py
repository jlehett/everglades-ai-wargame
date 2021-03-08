import argparse
import datetime
from functools import partial

import gym
import gym_everglades
from everglades_server import server
import neat
import numpy as np
import neat
import agents.NEAT.Visualize
import multiprocessing
from agents.State_Machine.random_actions_delay import random_actions

n = 1

test_n = 100
TEST_MULTIPLIER = 1
T_STEPS = 10000
TEST_REWARD_THRESHOLD = None

ENVIRONMENT_NAME = None
CONFIG_FILENAME = 'agents/NEAT/neat-config.txt'

NUM_WORKERS = multiprocessing.cpu_count()
CHECKPOINT_GENERATION_INTERVAL = 1
CHECKPOINT_PREFIX = None
GENERATE_PLOTS = True

PLOT_FILENAME_PREFIX = None
MAX_GENS = 1000
RENDER_TESTS = True

env = None

config = None

class NEATAgent():

    def _init_(self, max_gens, render_tests, check_point_interval):
        global MAX_GENS
        global RENDER_TESTS
        global CHECKPOINT_GENERATION_INTERVAL

        MAX_GENS = max_gens
        RENDER_TESTS = render_tests
        CHECKPOINT_GENERATION_INTERVAL = check_point_interval
        global n
        self.n = n
   
    def _eval_genomes(self,eval_single_genome, genomes, neat_config):
        parallel_evaluator = neat.ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

        parallel_evaluator.evaluate(genomes, neat_config)


    def _run_neat(self,checkpoint, eval_network, eval_single_genome):
        # Create the population, which is the top-level object for a NEAT run.

        self.print_config_info()

        if checkpoint is not None:
            print("Resuming from checkpoint: {}".format(checkpoint))
            p = neat.Checkpointer.restore_checkpoint(checkpoint)
        else:
            print("Starting run from scratch")
            p = neat.Population(config)

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(False))

        # Run until a solution is found.
        winner = p.run(partial(self._eval_genomes, eval_single_genome), n=MAX_GENS)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        net = neat.nn.FeedForwardNetwork.create(winner, config)

        self.test_genome(eval_network, net)

        self.generate_stat_plots(stats, winner)

        print("Finishing...")


    def generate_stat_plots(self,stats, winner):
        if GENERATE_PLOTS:
            print("Plotting stats...")
            agents.NEAT.Visualize.draw_net(config, winner, view=False, node_names=None, filename=PLOT_FILENAME_PREFIX + "net")
            agents.NEAT.Visualize.plot_stats(stats, ylog=False, view=False, filename=PLOT_FILENAME_PREFIX + "fitness.svg")
            agents.NEAT.Visualize.plot_species(stats, view=False, filename=PLOT_FILENAME_PREFIX + "species.svg")


    def test_genome(self,eval_network, net):
        reward_goal = config.fitness_threshold if not TEST_REWARD_THRESHOLD else TEST_REWARD_THRESHOLD

        print("Testing genome with target average reward of: {}".format(reward_goal))

        rewards = np.zeros(test_n)

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
        debug = False
        win_rate = 0

        for i in range(test_n * TEST_MULTIPLIER):

            print("--> Starting test episode trial {}".format(i + 1))
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
            reward = {0: 0, 1: 0}
            final_reward = 0
            t = 0

            reward_episode = 0

            while not done:

                if RENDER_TESTS:
                    env.render()

                observations, reward, done, info = env.step(actions)

                # print("\t Observation {}: {}".format(t, observation))
                # print("\t Info {}: {}".format(t, info))

                final_reward = reward[0]

                observations, reward, done, info = env.step(actions)

                # print("\t Reward {}: {}".format(t, reward))

                actions[0] = eval_network(net, observations[0])
                actions[1] = players[1].get_action(observations[1])

                # print("\t Reward {}: {}".format(t, reward))

                t += 1

                if done:
                    # Only provide rewards after game is over
                    # All fitness rewards must be non-negative for NEAT's maximization functions
                    # Since rewards are based on signed percent difference, just add 1
                    reward_episode += final_reward + 1

                    if reward[0] > reward[1]:
                        win_rate += 1
                    print("<-- Test episode done after {} time steps with reward {}".format(t + 1, reward_episode))
                    pass

            rewards[i % test_n] = reward_episode

            if i + 1 >= test_n:
                average_reward = np.mean(rewards)
                print("Average reward for episode {} is {}".format(i + 1, average_reward))
                if average_reward >= reward_goal:
                    print("Hit the desired average reward in {} episodes".format(i + 1))
                    break

        print("Final win rate is {}".format((win_rate / test_n)))

    def print_config_info(self):
        print("Running environment: {}".format(env.spec.id))
        print("Running with {} workers".format(NUM_WORKERS))
        print("Running with {} episodes per genome".format(n))
        print("Running with checkpoint prefix: {}".format(CHECKPOINT_PREFIX))
        print("Running with {} max generations".format(MAX_GENS))
        print("Running with test rendering: {}".format(RENDER_TESTS))
        print("Running with config file: {}".format(CONFIG_FILENAME))
        print("Running with generate_plots: {}".format(GENERATE_PLOTS))
        print("Running with test multiplier: {}".format(TEST_MULTIPLIER))
        print("Running with test reward threshold of: {}".format(TEST_REWARD_THRESHOLD))


    def _parse_args(self):
        global NUM_WORKERS
        global CHECKPOINT_GENERATION_INTERVAL
        global CHECKPOINT_PREFIX
        global n
        global GENERATE_PLOTS
        global MAX_GENS
        global CONFIG_FILENAME
        global RENDER_TESTS
        global TEST_MULTIPLIER
        global TEST_REWARD_THRESHOLD

        parser = argparse.ArgumentParser()

        parser.add_argument('--checkpoint', nargs='?', default=None,
                            help='The filename for a checkpoint file to restart from')

        parser.add_argument('--workers', nargs='?', type=int, default=NUM_WORKERS, help='How many process workers to spawn')

        parser.add_argument('--gi', nargs='?', type=int, default=CHECKPOINT_GENERATION_INTERVAL,
                            help='Maximum number of generations between save intervals')

        parser.add_argument('--test_multiplier', nargs='?', type=int, default=TEST_MULTIPLIER)

        parser.add_argument('--test_reward_threshold', nargs='?', type=float, default=TEST_REWARD_THRESHOLD)

        parser.add_argument('--checkpoint-prefix', nargs='?', default=CHECKPOINT_PREFIX,
                            help='Prefix for the filename (the end will be the generation number)')

        parser.add_argument('-n', nargs='?', type=int, default=n, help='Number of episodes to train on')

        parser.add_argument('--generate_plots', dest='generate_plots', default=False, action='store_true')

        parser.add_argument('-g', nargs='?', type=int, default=MAX_GENS, help='Max number of generations to simulate')

        parser.add_argument('--config', nargs='?', default=CONFIG_FILENAME, help='Configuration filename')

        parser.add_argument('--render_tests', dest='render_tests', default=False, action='store_true')

        command_line_args = parser.parse_args()

        NUM_WORKERS = command_line_args.workers

        CHECKPOINT_GENERATION_INTERVAL = command_line_args.gi

        CHECKPOINT_PREFIX = command_line_args.checkpoint_prefix

        CONFIG_FILENAME = command_line_args.config

        n = command_line_args.n

        MAX_GENS = command_line_args.g

        TEST_MULTIPLIER = command_line_args.test_multiplier

        TEST_REWARD_THRESHOLD = command_line_args.test_reward_threshold

        return command_line_args


    def run(self,eval_network, eval_single_genome, environment_name):
        global ENVIRONMENT_NAME
        global CONFIG_FILENAME
        global env
        global config
        global CHECKPOINT_PREFIX
        global PLOT_FILENAME_PREFIX

        ENVIRONMENT_NAME = environment_name

        env = gym.make(ENVIRONMENT_NAME)

        command_line_args = self._parse_args()

        checkpoint = command_line_args.checkpoint

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            CONFIG_FILENAME)

        if CHECKPOINT_PREFIX is None:
            timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
            CHECKPOINT_PREFIX = CONFIG_FILENAME.lower() + "_" + "cp_" + timestamp + "_gen_"

        if PLOT_FILENAME_PREFIX is None:
            timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
            PLOT_FILENAME_PREFIX = "plot_" + CONFIG_FILENAME.lower() + "_" + timestamp + "_"

        self._run_neat(checkpoint, eval_network, eval_single_genome)