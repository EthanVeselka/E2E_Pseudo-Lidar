import os
import csv
import torch
import numpy as np
import PIL
from torch.utils.data import Dataset
from processing import normalizer
from processing import readers

# import utils


# ---WARNING: INCOMPLETE---#
class PLDataset(Dataset):
    def __init__(self, root, n_samples, num_workers, seed, task):
        super(PLDataset, self).__init__(
            root, n_samples, num_workers, seed, task == "train"
        )
        self.n_samples = n_samples
        self.num_workers = num_workers
        self.seed = seed
        self.task = task

        self._read_data(root)
        self._load_data(n_samples)
        self._normalizer = Normalizer()

    def __getitem__(self, idx):
        # get item at idx, convert to tensor
        pass

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
        list_file = os.path.join(self.root, self.task + ".csv")

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
        what normalization is happening here?
        how does depth map model expect input?
        """
        pass
