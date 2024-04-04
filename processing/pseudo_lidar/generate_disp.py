import argparse
import os

import numpy as np
from PIL import Image

# import calib_utils
from . import calib_utils


# Generates ground truth disparities for training from LiDAR ground truths #


def generate_dispariy_from_velo(pc_velo, height, width, calib):
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
                    # print(curr_dir +"/"+ frame)
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
                    disp = generate_dispariy_from_velo(
                        point_cloud, height, width, calib
                    )
                    np.save("left_disp.npy", disp)
                    # print(width, height)
                    # im = Image.fromarray(disp)
                    # im = im.convert('RGB')
                    # im.save("left_disp_im.png")
                    # print(point_cloud.shape)
                    os.chdir("..")
    os.chdir("../../../../..")


# generate_disparity("carla_data/example_data")
