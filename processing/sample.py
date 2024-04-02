import random
import os
import torch
import numpy as np
import configparser
from ast import literal_eval


# ---WARNING: INCOMPLETE---#
def sample(root, config="config.ini", save_file_path="./output"):
    """
    Sample from root directory using config, returns listfiles for train/test splits

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    :param save_file_path: directory where listfiles will be saved
    :type save_file_path: str
    """

    print(os.getcwd())

    conf = configparser.ConfigParser()
    conf.read(os.path.join("processing",config))

    ALL = conf["General"]["ALL"]
    SPLITS = conf["General"]["SPLITS"]
    SAMPLE_SIZE = conf["General"]["SAMPLE_SIZE"]
    DATA_PATH = conf["Path"]["DATA_PATH"]
    EGO_BEHAVIOR = conf["Internal Variables"]["EGO_BEHAVIOR"]
    EXTERNAL_BEHAVIOR = conf["External Variables"]["EXTERNAL_BEHAVIOR"]
    WEATHER = conf["External Variables"]["WEATHER"]
    MAP = conf["External Variables"]["MAP"]

    DATA_PATH = os.path.join(os.getcwd(), DATA_PATH)
    SPLITS = literal_eval(SPLITS)
    #SPLITS = [float(x) for x in SPLITS]
    SAMPLE_SIZE = int(SAMPLE_SIZE)
    WEATHER = int(WEATHER)
    
    # Check values
    try:
        assert ALL in ["True", "False"]
        assert sum(SPLITS) == 1
        assert SAMPLE_SIZE > 0
        assert os.path.exists(DATA_PATH)
        if not ALL:
            assert EGO_BEHAVIOR in ["normal", "aggressive", "cautious"]
            assert EXTERNAL_BEHAVIOR in ["normal", "aggressive", "cautious"]
            assert WEATHER in [1, 2, 5, 8, 9, 12]
            assert MAP in ["Town01", "Town02", "Town07"]
    except AssertionError:
        print(SPLITS)
        raise ValueError("Invalid configuration parameters. Please check config.ini.")

    frames = []
    os.chdir(DATA_PATH)
    for episode in os.listdir(DATA_PATH):
        if episode == ".gitignore":
            continue
        if episode == "left_rgb_mat.csv":
            continue
        if episode == "calibmatrices.txt":
            continue
        os.chdir(episode)
        curr_dir = DATA_PATH + "/" + episode
        ep_conf = configparser.ConfigParser()
        ep_conf.read(os.path.join(curr_dir, config))
        ego_behav = ep_conf["Internal Variables"]["EGO_BEHAVIOR"]
        if not ALL and not (ego_behav == EGO_BEHAVIOR):
            os.chdir(DATA_PATH)
            continue

        for iteration in os.listdir(curr_dir):
            if iteration == config:
                continue
            os.chdir(iteration)
            curr_dir = DATA_PATH + "/" + episode + "/" + iteration
            
            for timestamp in os.listdir(curr_dir):
                if timestamp == config:
                    continue
                
                os.chdir(timestamp)
                curr_dir = DATA_PATH + "/" + episode + "/" + iteration + "/" + timestamp
                
                it_conf = configparser.ConfigParser()
                it_conf.read(config)

                if ALL or (
                    it_conf["External Variables"]["EXTERNAL_BEHAVIOR"] == EXTERNAL_BEHAVIOR
                    and it_conf["External Variables"]["WEATHER"] == WEATHER
                    and it_conf["External Variables"]["MAP"] == MAP
                ):
                    frames.extend([frame for frame in os.listdir(curr_dir) if frame != "config.ini"])

            
                
    # randomly sample indices from directories
    #print(range(len(frames)), curr_dir)
    random_indices = random.sample(range(len(frames)), SAMPLE_SIZE)
    random.shuffle(random_indices)
    
    train_len = int(SPLITS[0] * SAMPLE_SIZE)
    val_len = int(SPLITS[1] * SAMPLE_SIZE)

    train = random_indices[:train_len]
    val = random_indices[train_len : train_len + val_len]
    test = random_indices[train_len + val_len:]

    file = open(save_file_path + "/train.csv", "w")
    for idx in train:
        file.write(curr_dir + "/" + frames[idx] + "\n")
    file.close()

    if len(val) != 0:
        file = open(save_file_path + "/val.csv", "w")
        for idx in val:
            file.write(frames[idx] + "\n")
        file.close()

    file = open(save_file_path + "/test.csv", "w")
    for idx in test:
        file.write(frames[idx] + "\n")
    file.close()

    return save_file_path


# sample("../carla_data/data")
