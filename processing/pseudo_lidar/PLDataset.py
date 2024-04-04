import csv
import os
import torch
import torchvision.transforms as transforms
import random
from . import transforms as trf
from . import readpfm as rp
import numpy as np

from PIL import Image, ImageOps
from torch.utils.data import Dataset


def RGB_loader(path):
    return Image.open(path).convert("RGB")


# not sure how this changes for depth map
def disparity_loader(path):
    return np.load(path).astype(np.float32)


# ---WARNING: INCOMPLETE---# Refer to SceneFlowLoader.py or  for implementation
class PLDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        num_workers,
        seed,
        task,
        dploader=disparity_loader,
        rgbloader=RGB_loader,
        transform=None,
    ):
        # super(PLDataset, self).__init__(
        #     root, n_samples, num_workers, seed, task == "train"
        # )
        # self.n_samples = n_samples
        self.root = root
        self.num_workers = num_workers
        self.split_file = split_file
        self.seed = seed
        self.task = task
        self.transform = transform
        # if transform:
        #     self.transform = transform
        # else:
        #     self.transform = transforms.Compose([transforms.PILToTensor()])

        self.dploader = dploader
        self.rgbloader = rgbloader

        self.left_image_paths = []
        self.right_image_paths = []
        self.left_depths = []

        self._read_data()
        self.n_samples = len(self.left_image_paths)
        # self._load_data(n_samples)
        # self._normalizer = Normalizer()

    def __getitem__(self, idx):
        # get item at idx, convert to tensor

        # Get image data
        left_img = self.rgbloader(self.left_image_paths[idx])
        right_img = self.rgbloader(self.right_image_paths[idx])
        left_depth = self.dploader(self.left_depths[idx])

        if self.task == "train":
            left_img, right_img, left_depth = self._rand_crop(
                left_img, right_img, left_depth, 256, 512
            )  # parameters will need to be adjusted
        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 1200, h - 352, w, h))
            right_img = right_img.crop((w - 1200, h - 352, w, h))
            # left_depth = left_depth.crop((w - 1200, h - 352, w, h)) #for png
            left_depth = left_depth[h - 352 : h, w - 1200 : w]  # for numpy

        # Transform to tensor
        # transform = transforms.ToTensor()

        # left_img = transform(left_img)
        # right_img = transform(right_img)
        # left_depth = transform(left_depth)
        left_depth = torch.from_numpy(left_depth).float()

        # Additional transforms if necessary
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            
        processed = trf.get_transform(augment=False)  
        left_img = processed(left_img)
        right_img = processed(right_img)
        
        
        return left_img, right_img, left_depth

    def __len__(self):
        return self.n_samples

    def _read_data(self):
        # read data in chunks
        """
        read in chunks and index accordingly?
        how many images per iteration?
        """
        # left and right as x data --> convert to 2 3D tensors
        # depth_map ground truths as y data --> 2D tensors

        # self.split_file = os.path.join(self.root, self.split_file)
        list_file = os.path.join(self.split_file + "/output", self.task + ".csv")

        self.left_image_paths = []
        self.right_image_paths = []
        self.left_depths = []

        with open(list_file, "r+") as frame_path_folders:
            reader = csv.reader(frame_path_folders)
            next(reader, None)
            for row in reader:
                self.left_image_paths.append(self.root + row[0] + "/left_rgb.png")
                self.right_image_paths.append(self.root + row[0] + "/right_rgb.png")
                self.left_depths.append(
                    self.root + row[0] + "/left_disp.npy"
                )  # left_depth.png

    def _load_data(self, n_samples):
        # normalize data
        """
        what normalization is happening here? Prob rand transformations instead of norms, batchnorm looks to be done inside model itself
        how does disp model expect input?
        """
        pass

    def _rand_crop(self, l_img, r_img, dmap, th, tw):
        # from https://github.com/mileyan/pseudo_lidar/blob/master/psmnet/dataloader/KITTILoader.py

        w, h = l_img.size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        l_img = l_img.crop((x1, y1, x1 + tw, y1 + th))
        r_img = r_img.crop((x1, y1, x1 + tw, y1 + th))

        # dmap = dmap[y1 : y1 + th, x1 : x1 + tw]
        dmap = np.ascontiguousarray(dmap, dtype=np.float32) / 256
        dmap = dmap[y1 : y1 + th, x1 : x1 + tw]

        return l_img, r_img, dmap
