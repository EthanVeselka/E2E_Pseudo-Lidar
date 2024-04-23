import argparse
import os
import sys
import numpy as np
import csv


sys.path.append("../..")
BASE_DIR = "../.."
import processing.pseudo_lidar.calib_utils as calib_utils

# Generates psuedo-lidar point clouds from disparity | depth-map, used for Lidar 3D bb pred #


def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.50
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1.0 - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def project_depth_to_points(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Lidar")
    parser.add_argument("--root_dir", type=str, default="carla_data/data")
    parser.add_argument("--max_high", type=int, default=1)
    parser.add_argument("--is_depth", action="store_true")
    parser.add_argument("--calib_dir", type=str, default="carla_data/data")
    parser.add_argument(
        "--listfile_dir",
        type=str,
        default="carla_data/output",
    )

    args = parser.parse_args()
    args.listfile_dir = os.path.join(BASE_DIR, args.listfile_dir)
    args.calib_dir = os.path.join(BASE_DIR, args.calib_dir)
    args.root_dir = os.path.join(BASE_DIR, args.root_dir)

    assert os.path.isdir(args.listfile_dir)
    assert os.path.isdir(args.calib_dir)
    assert os.path.isdir(args.root_dir)

    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)

    calib_file = "{}/{}.txt".format(args.calib_dir, "calibmatrices")
    calib = calib_utils.Calibration(calib_file)

    for task in ["train", "test", "val"]:
        # print(os.getcwd())
        list_file = os.path.join(args.listfile_dir, task + ".csv")

        with open(list_file, "r+") as frame_path_folders:
            reader = csv.reader(frame_path_folders)
            # next(reader, None)

            for row in reader:
                path = os.path.join(args.root_dir, row[0], "output")
                if not os.path.exists(path):
                    os.mkdir(path)

                disp_map = np.load(os.path.join(path, "left_disp.npy"))
                disp_map = (disp_map * 256).astype(np.uint16) / 256.0
                lidar = project_disp_to_points(calib, disp_map, args.max_high)
                lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
                lidar = lidar.astype(np.float32)

                cd = os.getcwd()
                os.chdir(path)
                lidar.tofile("{}.bin".format("pseudo_lidar"))
                os.chdir(cd)
