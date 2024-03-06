import random
import os
import torch
import numpy as np

from processing import sample
from processing import normalizer
from processing import PLDataset
from processing import utils

from torch.utils.data import DataLoader


# ---WARNING: INCOMPLETE---#
# add parameters
def process(root, config):
    """
    Process sampled data, returns DataLoaders for PyTorch models

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    """

    save_file_path = sample(
        root, config, save_file_path="/output"
    )  # return file path with train/val/test listfiles
    # sample from clean data using sample.py and config
    # create PLDataset using sample data listfile (dataset creation does reading and normalization)
    # create DataLoaders from PLDataset to be used in models
    # return DataLoaders
    
    pass
