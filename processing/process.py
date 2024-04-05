import random
import os
import torch
import numpy as np

from . import sample
from . import utils
from .pseudo_lidar import PLDataset as pld
from .pseudo_lidar import generate_disp as gd
from torch.utils.data import DataLoader

# If you want to run this by itself, run from main directory and use:
# python3 -m processing.process


# ---WARNING: INCOMPLETE---#
# add parameters
def process(root, config, gen_disp=False):
    """
    Process sampled data, returns DataLoaders for PyTorch models

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    """
    save_file_path = os.getcwd() + "/carla_data/output"

    if gen_disp:
        gd.generate_disparity("carla_data/example_data")

    sample.sample(root, config, save_file_path)
    pldataset = pld.PLDataset(
        "carla_data/example_data",
        "carla_data/output",
        num_workers=4,
        seed=0,
        task="train",
    )
    pldl = DataLoader(pldataset, batch_size=64, shuffle=True)
    return pldl


if __name__ == "__main__":
    process("NA", "config.ini", True)
