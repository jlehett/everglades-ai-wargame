import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Discrete, Box

import everglades_server.server as server

try:
    from gym_everglades.envs.everglades_renderer import EvergladesRenderer
except:
    pass

import numpy as np
import pdb

MAX_SCORE = 3700

class EvergladesEnv(gym.Env):

    def __init__(self):
        # Game parameters
        self.num_turns = 150
        self.num_units = 100
        self.num_groups = 12
        self.num_nodes = 11
        self.num_actions_per_turn = 7
        self.unit_classes = ['controller', 'striker', 'tank']

        # Define the action space
        self.action_space = Tuple((Discrete(self.num_groups), Discrete(self.num_nodes + 1)) * self.num_actions_per_turn)

        # Define the state space
        self.observation_space = self._build_observation_space()

        return

    def step(self, actions):

        scores, status = self.game.game_turn(actions)
        observations = self._build_observations()

        reward = {i:0 for i in self.players}
        done = 0 
        if status != 0:
            done = 1
            if scores[0] != scores[1]:
                ### Boosted win score to compensate for new reward system
                reward[0] = 1 if scores[0] > scores[1] else 0
                ###

                reward[1] = 1 if scores[1] > scores[0] else -1
            # else reward is 0 for a tie
            #print(scores)
        # end status done check
        #print(status)
        else:
            ### Use the score calculated in server.game_end (used for non game ending rounds as well)
            # Score is calculated from number of nodes held for a given amount of time
            # Should incentivize the agent to be more agressive
            # Scores range from 0- ~3000 points
            # May want to attempt normalization in the future (make sure to change win score back to 1 if so)
            # Normalized by MAX_SCORE
            # Subtraction puts emphasis on having more points than the other player. Should force agent to always try and take
            # Another objective if its score is negative
            reward[0] = scores[0] / MAX_SCORE
            #reward_0 = reward[0]
            #reward[1] = scores[1] / MAX_SCORE
            #reward[0] -= reward[1]
            #reward[1] -= reward_0
            ######################################################

            # Percent Difference Reward
            #reward[0] = (scores[0]-scores[1]) / ((scores[0] + scores[1]) / 2)
            #reward[1] = (scores[1]-scores[0]) / ((scores[0] + scores[1]) / 2)

            #if reward[0] < 0:
            #    reward[0] = 0

        # return state, reward, done, info
        return observations, reward, done, {}

    def reset(self, **kwargs):
    # kwargs is allowed. https://github.com/openai/gym/blob/master/gym/core.py
        # Get Players
        self.players = kwargs.get('players')
        config_dir = kwargs.get('config_dir')
        map_file = kwargs.get('map_file')
        unit_file = kwargs.get('unit_file')
        output_dir = kwargs.get('output_dir')
        player_names = kwargs.get('pnames')
        self.debug = kwargs.get('debug',False)

        # Input validation
        assert( len(self.players) == 2 ), 'Must have exactly two players' # for now
        self.pks = self.players.keys()
        self.sorted_pks = sorted(self.pks)
        self.player_dat = {}
        for i in self.pks:
            self.player_dat[i] = {}
            self.player_dat[i]['unit_config'] = self._build_groups(i)

        # Initialize game
        self.game = server.EvergladesGame(
                config_dir = config_dir,
                map_file = map_file,
                unit_file = unit_file,
                output_dir = output_dir,
                pnames = player_names,
                debug = self.debug
        )
        
        # Initialize players with selected groups
        self.game.game_init(self.player_dat)

        try:
            self.renderer = EvergladesRenderer(self.game)
        except:
            pass

        # Get first game state
        observations = self._build_observations()
        #pdb.set_trace()

        return observations

    def render(self, mode='human'):
        self.renderer.render(mode)

    def close(self):
        self.renderer.close()

    def _build_observation_space(self):
        group_low = np.array([1, 0, 0, 0, 0])  # node loc, class, avg health, in transit, num units rem
        group_high = np.array([self.num_nodes, len(self.unit_classes), 100, 1, self.num_units])

        group_state_low = np.tile(group_low, self.num_groups)
        group_state_high = np.tile(group_high, self.num_groups)

        control_point_portion_low = np.array([0, 0, -100, -1])  # is fortress, is watchtower, percent controlled, num opp units
        control_point_portion_high = np.array([1, 1, 100, self.num_units])

        control_point_state_low = np.tile(control_point_portion_low, self.num_nodes)
        control_point_state_high = np.tile(control_point_portion_high, self.num_nodes)

        observation_space = Box(
            low=np.concatenate([[1], control_point_state_low, group_state_low]),
            high=np.concatenate([[self.num_turns + 1], control_point_state_high, group_state_high])
        )
        #pdb.set_trace()

        return observation_space

    def _build_groups(self, player_num):
        unit_configs = {}

        # In the future, get group assignments from the agent as a kwarg in .reset()
        num_units_per_group = int(self.num_units / self.num_groups)
        for i in range(self.num_groups):
            unit_type = self.unit_classes[i % len(self.unit_classes)]
            if i == self.num_groups - 1:
                unit_configs[i] = (unit_type, self.num_units - sum([c[1] for c in unit_configs.values()]))
            else:
                unit_configs[i] = (unit_type, num_units_per_group)
        return unit_configs

    def _build_observations(self):
        observations = {}

        for player in self.players:
            board_state = self.game.board_state(player)
            player_state = self.game.player_state(player)

            state = np.zeros(board_state.shape[0] + player_state.shape[0] - 1)
            state[0:board_state.shape[0]] = board_state
            state[board_state.shape[0]:] = player_state[1:]

            observations[player] = state

        return observations

# end class EvergladesEnv

if __name__ == '__main__':
    test_env = EvergladesEnv()
