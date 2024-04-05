import argparse
import os
import sys

import numpy as np
from PIL import Image

# import calib_utils
sys.path.append("../..")
import processing.pseudo_lidar.calib_utils as calib_utils


# Generates ground truth disparities for training from LiDAR ground truths #


def generate_disparity_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < width - 1)
        & (pts_2d[:, 0] >= 0)
        & (pts_2d[:, 1] < height - 1)
        & (pts_2d[:, 1] >= 0)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    baseline = 0.5

    disp_map = (calib.f_u * baseline) / depth_map
    return disp_map


def generate_disparity(filepath):

    config = "config.ini"
    calib = calib_utils.Calibration(filepath + "/calibmatrices.txt")
    filepath = os.path.join(os.getcwd(), filepath)
    os.chdir(filepath)

    for episode in os.listdir(filepath):
        if episode == ".gitignore":
            continue
        if episode == "calibmatrices.txt":
            continue

        os.chdir(episode)
        curr_dir = filepath + "/" + episode

        for iteration in os.listdir(curr_dir):
            if iteration == config:
                continue
            os.chdir(iteration)
            curr_dir = filepath + "/" + episode + "/" + iteration

            for timestamp in os.listdir(curr_dir):
                if timestamp == config:
                    continue

                os.chdir(timestamp)
                curr_dir = filepath + "/" + episode + "/" + iteration + "/" + timestamp

                for frame in os.listdir(curr_dir):
                    if frame == "config.ini":
                        continue
                    os.chdir(frame)
                    lidar = open("left_lidar.ply", "r+")
                    lines = lidar.readlines()
                    lines = lines[10:]

                    points = []
                    for line in lines:
                        values = line.split()[:3]  # Extract the first 3 values

                        point = [
                            float(value) for value in values
                        ]  # Convert values to floats
                        points.append(point)

                    print("s:", len(points), frame)
                    point_cloud = np.array(points)
                    img = Image.open("left_rgb.png")
                    width, height = img.size
                    disp = generate_disparity_from_velo(
                        point_cloud, height, width, calib
                    )
                    if not os.path.exists("output"):
                        os.mkdir("output")
                    np.save("output/left_disp.npy", disp)

                    os.chdir("..")
    os.chdir("../../../../..")
