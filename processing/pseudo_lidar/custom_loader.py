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


def dataloader(datapath, split_file, task="train"):

    left_image_paths = []
    right_image_paths = []
    left_disps = []

    if task == "train":
        tasks = ["train"]
    elif task == "val":
        tasks = ["val"]
    elif task == "test":
        tasks = ["test"]
    else:
        tasks = ["train", "val", "test"]

    for tsk in tasks:
        list_file = os.path.join(split_file, tsk + ".csv")

        with open(list_file, "r+") as frame_path_folders:
            reader = csv.reader(frame_path_folders)
            next(reader, None)
            for frame in reader:
                left_image_paths.append(datapath + "/" + frame[0] + "/left_rgb.png")
                right_image_paths.append(datapath + "/" + frame[0] + "/right_rgb.png")
                left_disps.append(
                    datapath + "/" + frame[0] + "/output/left_disp.npy"
                )  # left_disp.png

    return left_image_paths, right_image_paths, left_disps
