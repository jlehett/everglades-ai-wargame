import pickle
import os

from utils.Agent_Loader.Saved_Agent_Types.smart_state_fixed_fc1 import SmartStateFixedFC1
from utils.Agent_Loader.Saved_Agent_Types.smart_state_fixed_fc2 import SmartStateFixedFC2

class AgentLoader:
    """
    Class for helping load agents for evaluation purposes.
    """
    def __init__(self, map_name):
        self.map_name = map_name

    """
    PUBLIC FUNCTIONS
    """

    def loadAgent(self, save_file_path, player_num):
        """
        Loads the specified agent save while utilizing the "type" property to
        determine how to load that specific agent.

        @param {string} save_file_path The path to the saved agent model
        @returns {Class Instance} Returns a class instance of the agent ready to be evaluated with the saved model
        """
        # Define agent_class_mapping like a switch statement mapping type to a specific loading function
        agent_class_mapping = {
            'Smart State Fixed 1 Hidden Layer': SmartStateFixedFC1,
            'Smart State Fixed 2 Hidden Layers': SmartStateFixedFC2
        }

        # If the save file does not exist, return early; nothing can be done
        if not os.path.exists(save_file_path + '.pickle'):
            print('Save file does not exist; exiting...')
            return None

        # Load the saved model
        save_file = open(save_file_path + '.pickle', 'rb')
        save_file_data = pickle.load(save_file)
        save_file.close()

        # If the `type` key does not exist in the save file, return early; nothing can be done
        if not 'type' in save_file_data:
            print('Save file does not have a `type` property; exiting...')
            return None

        # Grab the agent type from the save file
        agent_type = save_file_data['type']

        # If the type doesn't match one of the specified mappings, return early; nothing can be done
        if not agent_type in agent_class_mapping:
            print('Agent type mapping to an agent class does not exist; exiting...')
            return None

        # Initialize the appropriate class
        agent_class_instance = agent_class_mapping[agent_type](save_file_data, player_num, self.map_name)

        # Return the loaded instance
        return agent_class_instance
