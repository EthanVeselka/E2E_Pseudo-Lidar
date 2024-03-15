import csv
import os
import torch
import torchvision.transforms as transforms
import random
import transforms as trf
import readpfm as rp
import numpy as np

from PIL import Image, ImageOps
from torch.utils.data import Dataset


# ---WARNING: INCOMPLETE---# Refer to SceneFlowLoader.py or  for implementation
class PLDataset(Dataset):
    def __init__(self, root, n_samples, num_workers, seed, task, transform=None):
        super(PLDataset, self).__init__(
            root, n_samples, num_workers, seed, task == "train"
        )
        self.n_samples = n_samples
        self.num_workers = num_workers
        self.seed = seed
        self.task = task
        self.transform = transform

        self.left_image_paths = []
        self.right_image_paths = []
        self.left_depths = []

        self._read_data(root)
        # self._load_data(n_samples)
        self._normalizer = Normalizer()

    def __getitem__(self, idx):
        # get item at idx, convert to tensor

        # Get image data
        left_rgb = Image.open(self.left_image_paths[idx])
        right_rgb = Image.open(self.right_image_paths[idx])
        left_depth = Image.open(self.left_depths[idx])

        # Transform it if necessary (ToTensor(), etc...)
        if self.transform:
            left_rgb = self.transform(left_rgb)
            right_rgb = self.transform(right_rgb)
            left_depth = self.transform(left_depth)

        return left_rgb, right_rgb, left_depth

    def __len__(self):
        return self.n_samples

    def _read_data(self, root):
        # read data in chunks
        """
        read in chunks and index accordingly?
        how many images per iteration?
        """
        # left and right as x data --> convert to 2 3D tensors
        # depth_map ground truths as y data --> 2D tensors
        list_file = os.path.join(root, self.task + ".csv")

        self.left_image_paths = []
        self.right_image_paths = []
        self.left_depths = []

        with open(list_file, "r+") as frame_path_folders:
            reader = csv.reader(frame_path_folders)
            next(reader, None)

            for row in reader:
                self.left_image_paths.append(row + "/left_rgb.png")
                self.right_image_paths.append(row + "/right_rgb.png")
                self.left_depths.append(row + "/left_depth.png")

    def _load_data(self, n_samples):
        # normalize data
        """
        what normalization is happening here? Prob rand transformations instead of norms, batchnorm looks to be done inside model itself
        how does disp model expect input?
        """
        pass
