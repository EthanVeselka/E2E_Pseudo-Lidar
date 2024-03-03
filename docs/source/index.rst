
Psuedo-Lidar Documentation
=======================================

This project implements Pseudo-Lidar using CARLA data.


Config Files
------------

Config files are used to specify parameters for the models and processing.


CARLA Data
----------

This library contains CARLA datasets that can be used in processing for training and testing models.


Processing
----------

Data processing is done here, including cleaning, flagging for occlusion, and sampling training and testing data to create PyTorch DataLoaders used by the models. Use the config file in `E2E_Pseudo-Lidar.processing` 
to specify parameters for creating training and testing sets.


Models
------

The three pipeline models are contained here. The first is the depth estimator, which creates a depth estimation map, second is the pseudo-lidar projection, and last is a given variation of Lidar algorithm.
        

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
--------

.. toctree::

   Config
   CARLA
   Processing
   Models