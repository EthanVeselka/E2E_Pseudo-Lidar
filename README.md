Implementing the Pseudo-LiDAR framework from "Pseudo-LiDAR from Visual Depth Estimation:
Bridging the Gap in 3D Object Detection for Autonomous Driving" for self-driving in PyTorch, using CARLA and OpenCV.
# Command-Line Interface
Usage: `python PL_cli [-h] [--key KEY] [--value VALUE] [--file-path FILE_PATH]`

## Universal Options
These options are found in every config file.

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
- Description: Time/Weather preset. Must select a preset from the list below.
- Options: 1 (ClearNoon), 2 (CloudyNoon), 5 (MidRainyNoon), 8 (ClearSunset), 9 (CloudySunset), 12 (MidRainSunset)

**MAP** : str
- Description: CARLA town map. Must select a map from the list below.
- Options: Town01, Town02, Town07
*Links:*
- https://carla.readthedocs.io/en/latest/map_town01/
- https://carla.readthedocs.io/en/latest/map_town02/
- https://carla.readthedocs.io/en/latest/map_town07/

## Sampling Options
The `processing/config.ini` file contains these parameters for sampling from the CARLA data, which is then fed to the model. 

**ALL** : bool
- Description: Sample from all internal and external configurations.

**SPLITS** : 3-tuple of floats
- Format: "(train, validation, test)". **Must include parentheses and commas.**
- Description: Proportion of set to be used for train/val/test splits, must sum to 1.

**SAMPLE_SIZE** : int
- Description: Number of frames to sample for train/val/test sets.

## CARLA Data Options
The `carla_data/config.ini` file contains these parameters for 

**CARLA_PYTHON_PATH** : str
- Description : Absolute path to CARLA Python API.

**POLL_RATE** : float
- Description : Rate at which sensors will output data per second.

**CAMERA_X** : int
- Description : Horizontal camera resolution.

**CAMERA_Y** : int
- Description : Vertical camera resolution.

**CAMERA_FOV** : int
- Description : Field of view of camera in degrees.
