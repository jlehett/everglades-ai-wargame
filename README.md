# Installation Instructions
## Dependencies
Everglades runs in a Python3 environment. Ensure the python packages ***gym*** and ***numpy*** are installed. This can be done with:
```bash
$ pip install numpy
$ pip install gym
```
If your computing environment requires it, make sure to include the --cert and --proxy flags with the pip commands.

## Installation
From the root Everglades directory, install the Everglades environment with:
```bash
pip install -e gym-everglades/
```
Next, install the Everglades server with:
```bash
pip install -e everglades-server/
```

# File and Directory Descriptions

### ./agents/

This is a common directory where any created agents for the Everglades game can be stored. Some example files are included with the package.

### ./config/

This directory containes setup files which are used for game logic. Currently only the DemoMap.json and UnitDefinitions.json files are used for gameplay. They can be swapped for files defining a different map or units, but note that any swaps likely will cause inflexible server logic to break.

### ./everglades-server/

This directory contains the main logic for the Everglades game. 

### ./gym-everglades/

This directory is the OpenAI Gym for project Everglades. It follows the Gym API standards.

### ./README.md

This file, explaining important directory structure and installation requirements.

### ./.gitignore

This file tells git to ignore compiled files and telemetry output. 