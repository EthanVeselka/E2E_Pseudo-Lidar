# E2E_Pseudo-Lidar
Implementing an end-to-end Pseudo-LiDAR model framework for autonomous driving in the CARLA simulator.

# Command-Line Interface
Usage: `python .\PL_cli\ [-h] [--key KEY] [--value VALUE]`
You must be in the `PL_cli` directory to run the CLI. Note that there is another subdirectory called `PL_cli` within the `PL_cli` directory; run the commands from the outermost `PL_cli` directory.

## Data Collection Options
The `processing/config.ini` file can be modified through the PL_CLI. This file contains the parameters for CARLA data collection, which is then fed to the model. 

**ALL** : bool
- Description: Sample from all internal and external configurations.

**SPLITS** : 3-tuple of floats
- Format: "train, validation, test"
- Must include parentheses and commas.
- Description: Proportion of set to be used for train/val/test splits, must sum to 1.

**SAMPLE_SIZE** : int
- Description: Number of frames to sample for train/val/test sets.

**DATA_PATH** : str 
- Default: /E2E_Pseudo-Lidar/carla_data/data
- Description: Path to data; this is the default root directory.

**EGO_BEHAVIOR** : str
- Description: Internal driver style.
- Options: normal, aggressive, cautious

**EXTERNAL_BEHAVIOR** : str
- Description: External driver style.
- Options: normal, aggressive, cautious

**WEATHER** : int
- Description: Time/Weather preset; e.g. 1 = ClearNoon.
- Options: 1, 2, 5, 8, 9, 12

**MAP** : str
- Description: CARLA town map.
- Options: Town01, Town02, Town07