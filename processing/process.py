import random
import os
import torch
import numpy as np

from . import sample
from . import utils
from . import generate_disp
from .dataLoading import PLDataset
from .dataLoading import custom_loader

from torch.utils.data import DataLoader

#If you want to run this by itself, run from main directory and use:
# python3 -m processing.process

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
    save_file_path = os.getcwd() + "/carla_data/output"
    
    save_file_path = sample.sample(
        root, config, save_file_path
    )  # return file path with train/val/test listfiles

    # generate true disp from lidar truths (need for training disp model, only if training instead of using pretrained)
    # create PLDataset using sample data listfile (dataset creation does reading and transformations)
    # create torch DataLoaders using custom_loader from PLDataset
    # return DataLoaders
    
    
    pldataset = PLDataset.PLDataset(save_file_path, num_workers=4, seed=0, task="train") #num_samples 
    
    pldl = DataLoader(pldataset, batch_size = 64, shuffle = True)
    
    #test loop
    for batch in pldl:
        for images in batch:
            print(images.shape)
    
    return pldl

if __name__ == "__main__":
    process("NA", "config.ini")