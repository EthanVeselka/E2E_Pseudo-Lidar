import os
import csv

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(datapath, split_file):

    list_file = os.path.join(split_file, "train.csv")

    left_image_paths = []
    right_image_paths = []
    left_disps = []

    with open(list_file, "r+") as frame_path_folders:
        reader = csv.reader(frame_path_folders)
        next(reader, None)
        for row in reader:
            left_image_paths.append(datapath + "/" + row[0] + "/left_rgb.png")
            right_image_paths.append(datapath + "/" + row[0] + "/right_rgb.png")
            left_disps.append(
                datapath + "/" + row[0] + "/output/left_disp.npy"
            )  # left_disp.png

    return left_image_paths, right_image_paths, left_disps
