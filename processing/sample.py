import random
import os
import torch
import numpy as np


# ---WARNING: INCOMPLETE---#
def sample(root, config, save_file_path="/output"):
    """
    Sample from root directory using config, returns listfiles for train/test splits

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    :param save_file_path: directory where listfiles will be saved
    :type save_file_path: str
    """

    # select indices/filter clean data according to config
    # create sample listfile (dictionary of files to be read by readers)
    # save to file path
    pass
