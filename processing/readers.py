import os
import random
import numpy as np


# ---WARNING: INCOMPLETE---#
class Reader:
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir
        self._current_index = 0

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        # read example at index
        pass

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class SampleReader(Reader):
    # this is for reading data in chunks from sampled data to PLDataset for models
    # normalize data here
    pass
