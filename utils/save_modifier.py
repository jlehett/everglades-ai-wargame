import pickle
import os

def modifySaveFile(save_file_path, modifications):
    """
    Function to apply new data to an existing save file.

    @param {string} save_file_path The path to the save file to modify
    @param {dict} modifications The modifications to make to the save file
    """
    # If the file does not exist, return early; nothing can be done
    if not os.path.exists(save_file_path + '.pickle'):
        print('Save file does not exist; exiting...')
        return

    # Load the save file to edit
    save_file = open(save_file_path + '.pickle', 'rb')
    save_file_data = pickle.load(save_file)
    save_file.close()

    # Add the specified data to the dictionary
    for key in modifications.keys():
        save_file_data[key] = modifications[key]

    # Save the file
    save_file = open(save_file_path + '.pickle', 'wb')
    pickle.dump(save_file_data, save_file)
    save_file.close()

    print('Successfully saved modifications; exiting...')
    

# Logic to run if this file itself is run
if __name__ == '__main__':
    # Variables to modify to use the script
    SAVE_FILE_PATH='saved-agents/self_play'
    PROPERTY_MODIFICATIONS = {
        'type': 'Smart State Fixed 2 Hidden Layers',
        'description': 'Smart State with 2 hidden layers of size 80. Trained in self_play_royale.',
        'reward type': 'Transition - Normalized Score to Reward Short Games',
        'keys to display': ['type', 'description',  'reward type', 'episodes', 'fc1_size', 'fc2_size', 'batch_size', 'target_update', 'memory_size', 'gamma', 'n_step'],
    }
    
    # Modify the file
    modifySaveFile(SAVE_FILE_PATH, PROPERTY_MODIFICATIONS)
