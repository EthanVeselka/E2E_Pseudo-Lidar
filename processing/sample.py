import random
import os
import torch
import numpy as np
import configparser


# ---WARNING: INCOMPLETE---#
def sample(root, sample_size, config="config.ini", save_file_path="/output"):
    """
    Sample from root directory using config, returns listfiles for train/test splits

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    :param save_file_path: directory where listfiles will be saved
    :type save_file_path: str
    """

    conf = configparser.ConfigParser()
    conf.read(config)

    ALL = conf["General"]["ALL"]
    SPLITS = conf["General"]["SPLITS"]
    DATA_PATH = conf["Path"]["DATA_PATH"]
    EGO_BEHAVIOR = conf["Internal Variables"]["EGO_BEHAVIOR"]
    EXTERNAL_BEHAVIOR = conf["External Variables"]["EXTERNAL_BEHAVIOR"]
    WEATHER = conf["External Variables"]["WEATHER"]
    MAP = conf["External Variables"]["MAP"]

    # Check values
    assert sum(SPLITS) == 1

    frames = []
    os.chdir(DATA_PATH)
    for episode in os.listdir(DATA_PATH):
        os.chdir(episode)
        curr_dir = DATA_PATH + "/" + episode
        ep_conf = configparser.ConfigParser()
        ep_conf.read(config)
        ego_behav = ep_conf["Internal Variables"]["EGO_BEHAVIOR"]
        if not ALL and not (ego_behav == EGO_BEHAVIOR):
            os.chdir(DATA_PATH)
            continue

        for iteration in episode:
            if iteration == config:
                continue
            os.chdir(iteration)
            curr_dir = DATA_PATH + "/" + episode + "/" + iteration
            it_conf = configparser.ConfigParser()
            it_conf.read(config)

            if ALL or (
                it_conf["External Variables"]["EXTERNAL_BEHAVIOR"] == EXTERNAL_BEHAVIOR
                and it_conf["External Variables"]["WEATHER"] == WEATHER
                and it_conf["External Variables"]["MAP"] == MAP
            ):
                frames.extend([frame for frame in os.listdir(curr_dir)])

    # randomly sample indices from directories
    random_indices = random.sample(range(len(frames)), sample_size)
    random.shuffle(random_indices)

    train = random_indices[: SPLITS[0]]
    val = random_indices[SPLITS[0] : SPLITS[0] + SPLITS[1]]
    test = random_indices[SPLITS[0] + SPLITS[1] :]

    file = open(save_file_path + "/train", "w")
    for idx in train:
        file.write(frames[idx] + "\n")
    file.close()

    if len(val) != 0:
        file = open(save_file_path + "/val", "w")
        for idx in val:
            file.write(frames[idx] + ",\n")
        file.close()

    file = open(save_file_path + "/test", "w")
    for idx in test:
        file.write(frames[idx] + "\n")
    file.close()

    return save_file_path


# sample("../carla_data/data")
