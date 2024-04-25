Collection of scripts to allow for easy data collection and viewing for computer vision tasks in CARLA
# Files Included
**config.ini**\
This is the main file used in carla_data to specify hyperparameters that can be set before a simulation run.
The values included here are used in carla_client.py and data_viwer.py to set various hyperparameters. These
values can be set either by modifying the file directly, or by using the PL_cli script to edit the parameters
from the command line.

**carla_client.py**\
This is the main file used to run a simulation epoch. This script will read the parameters set in config.ini, and
run set the specified paramters to the CARLA server. Before running, first make sure that your CARLA server is running.
In the current implementation, carla_client.py will gather data from a Semantic Lidar sensor, two camera sensors, and save
the bounding boxes of various objects. These data will be saved to the save directory specified in the config.ini

**data_viewer.py**\
This script allows the user to view data for a specific run of data

## Scripts modified from CARLA example scripts

**automatic_control.py**
This script was taken and modified from the CARLA example scripts, where some functions were used in other scripts in /carla_data

**automatic_control.py**
This script was taken and modified from the CARLA example scripts, where some functions were used in other scripts in /carla_data

# carla_client.py:
This is the main file used to run a simulation epoch. This script will read the parameters set in config.ini, and
run set the specified paramters to the CARLA server. Before running, first make sure that your CARLA server is running.
In the current implementation, carla_client.py will gather data from a Semantic Lidar sensor, two camera sensors, and save
the bounding boxes of various objects. These data will be saved to the save directory specified in the config.ini

## Dependencies

**Pygame**
**Carla** - The Python API for CARLA
**Numpy**

## Simulation workflow

### main()

initial connection to the CARLA server is established, and vehicles and pedestrians are spawned and given controllers

### prep_episde()

Ego-car specific elements are set.
**Ego-car Behavior is set**
**Ego-car Sensors are set**
**Pygame window is created**

### sim_episode()

Implements the main simulation loop. Data will be collected into memory during the simulation, and writen to file once
the data is collected

# carla_client.py point-calibration functions

## build_projection_matrix
Returns a projection matrix for a camera, given the camera's width, heigth, and fov

*Arguments*:

**w** int | Width of the camera's resolution 

**h** int | Height of the camera's resolution

**fov** int, float | Camera's horizontal fov, given in degrees

## get_image_point
Returns the 2D position of an object with respect to where it would be on a camera's image.

*Arguments*:

**loc** Carla.location | location of point in world-space

**K** numpy.Array (3x3)| Projection matrix of the desired camera

**w2c** numpy.Array (3x3)| Transformation matrix for world points into camera points (often from data.transform.get_inverse_matrix())

## get_camera_point
Returns the 3D position of an object with respect to the camera. This returns in the Kitti Coordinate system, where +x will be to the right, +y will be down,
and +z will be forward depth.

*Arguments*:

**loc** Carla.location | location of point in world-space

**w2c** numpy.Array (3x3)| Transformation matrix for world points into camera points (often from data.transform.get_inverse_matrix())


# data_viewer.py:
This script allows the user to view data for a specific run of data. When ran, it will use the parameters specified in config.py
to determine where the data to be viewed is stored. 

## Dependencies
**Pygame**
**Numpy**

## Controls:
**Q/R** - Toggle bounding box type
**B** - Toggle 2D and 3D bounding boxes
**Up Arrow / Down Arrow** - Toggle viweing camera
**Left Arrow / Right Arrow** - View previous / next frame
