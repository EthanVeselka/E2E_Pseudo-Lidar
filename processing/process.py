import random
import os
import torch
import numpy as np

from processing import sample
from processing import utils
from processing import generate_disp
from dataLoading import PLDataset
from dataLoading import custom_loader
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

    # generate true disp from lidar truths (need for training disp model, only if training instead of using pretrained)
    # create PLDataset using sample data listfile (dataset creation does reading and transformations)
    # create torch DataLoaders using custom_loader from PLDataset
    # return DataLoaders

    pass
