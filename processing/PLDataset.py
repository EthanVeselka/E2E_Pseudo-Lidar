import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset


import Normalizer
import Reader
import utils


# ---WARNING: INCOMPLETE---#
class PLDataset(Dataset):
    def __init__(self, root, n_samples, num_workers, seed, train=False):
        super(PLDataset, self).__init__(root, n_samples, num_workers, seed, train)

        self.n_samples = n_samples
        self.num_workers = num_workers
        self.seed = seed
        self.train = train

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
        pass

    def _load_data(self, n_samples):
        # normalize data
        pass
