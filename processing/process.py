import random
import os
import torch
import numpy as np

import Cleaner
import Normalizer
import PLDataset
import utils

from torch.utils.data import DataLoader


# ---WARNING: INCOMPLETE---#
# add parameters
def process(root, config, clean=False):
    # clean data using Cleaner() if clean
    # sample from clean data using sample.py and config
    # create PLDataset using sample data listfile (dataset creation does reading and normalization)
    # create DataLoaders from PLDataset to be used in models
    # return DataLoaders
    pass
