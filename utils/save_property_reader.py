import os
import pickle


def displaySaveFileInfo(save_file_path):
    """
    Function to load and display the keys specified by the `keys to display` property
    in the save file.

    @param {string} save_file_path The path to the save file to load and display
    """
    # If the file does not exist, return early; nothing can be done
    if not os.path.exists(save_file_path + '.pickle'):
        print('Save file does not exist; exiting...')
        return

    # Load the save file to edit
    save_file = open(save_file_path + '.pickle', 'rb')
    save_file_data = pickle.load(save_file)
    save_file.close()

    # Check if the `keys to display` property exists in the save data
    if not save_file_data['keys to display']:
        print('Save file does not have a `keys to display` property; exiting...')
        return
    
    # If the save data has a `keys to display` property, then display those keys
    for key in save_file_data['keys to display']:
        if not key in save_file_data:
            value = '-'
        else:
            value = save_file_data[key]
        print('{}: {}'.format(key, value))

def displayAllKeys(save_file_path):
    """
    Function to load and display all key names the save file has in case you don't
    know what you're looking for.

    @param {string} save_file_path The path to the save file to load and display the keys of
    """
    # If the file does not exist, return early; nothing can be done
    if not os.path.exists(save_file_path + '.pickle'):
        print('Save file does not exist; exiting...')
        return

    # Load the save file to edit
    save_file = open(save_file_path + '.pickle', 'rb')
    save_file_data = pickle.load(save_file)
    save_file.close()
    
    # If the save data has a `keys to display` property, then display those keys
    print(save_file_data.keys())

def displaySpecifiedKeys(save_file_path, keys_to_display):
    """
    Function to load and display the specified keys with their values from a
    save file, in case the file does not have the `keys to display` property yet.

    @param {string} save_file_path The path to the save file to load and display the keys of
    @param {string[]} keys_to_display The keys to display from the save file
    """
    # If the file does not exist, return early; nothing can be done
    if not os.path.exists(save_file_path + '.pickle'):
        print('Save file does not exist; exiting...')
        return

    # Load the save file to edit
    save_file = open(save_file_path + '.pickle', 'rb')
    save_file_data = pickle.load(save_file)
    save_file.close()
    
    # For each of the specified keys, display the key and value if it exists
    for key in keys_to_display:
        if not key in save_file_data:
            value = '-'
        else:
            value = save_file_data[key]
        print('{}: {}'.format(key, value))


# Logic to run if this file itself is run
if __name__ == '__main__':
    # Variables to modify to run this script
    SAVE_FILE_PATH = 'saved-agents/smart_state_newton'
    
    # Call the displaySaveFileInfo function
    #displaySaveFileInfo(SAVE_FILE_PATH)

    # Call the displayAllKeys function
    displayAllKeys(SAVE_FILE_PATH)

    # Call the displaySpecifiedKeys function
    #displaySpecifiedKeys(SAVE_FILE_PATH, ['fc1_size', 'gamma'])