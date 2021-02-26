import json


class constants:
    def __init__(self, constants_path):
        with open(constants_path, 'r') as f:
            constants = json.load(f)

        # game setup
        self.player0 = constants['game_setup']['player0']
        self.player1 = constants['game_setup']['player1']
        self.n_episodes = constants['game_setup']['n_episodes']

        # memory
        self.network_save_name = constants['memory']['network_save_name']
        self.network_load_name = constants['memory']['network_load_name']
        self.network_save_after = constants['memory']['network_save_after']
        self.data_name = constants['memory']['data_name']

        # hyperparameters
        self.epsilon_start = constants['hyperparameters']['epsilon_start']
        self.epsilon_final = constants['hyperparameters']['epsilon_final']
        self.epsilon_decay = constants['hyperparameters']['epsilon_decay']
        self.input_size = constants['hyperparameters']['input_size']
        self.output_size = constants['hyperparameters']['output_size']
        self.fc1_size = constants['hyperparameters']['fc1_size']
        self.batch_size = constants['hyperparameters']['batch_size']
        self.target_update = constants['hyperparameters']['target_update']
        self.memory_size = constants['hyperparameters']['memory_size']
        self.epsilon_decay = constants['hyperparameters']['epsilon_decay']
        self.gamma = constants['hyperparameters']['gamma']
        self.learning_rate = constants['hyperparameters']['learning_rate']
        self.n_step = constants['hyperparameters']['n_step']

        # environment
        self.env_name = constants['environment']['env_name']
        self.config_dir = constants['environment']['config_dir']
        self.map_file = self.env_constants + constants['environment']['map_file']
        self.setup_file = self.env_constants + constants['environment']['setup_file']
        self.unit_file = self.env_constants + constants['environment']['unit_file']
        self.env_output_dir = constants['environment']['env_output_dir']
        self.debug = False