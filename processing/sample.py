import random
import os
import torch
import numpy as np
import configparser
from ast import literal_eval

BASE_DIR = ".."
# sys.path.append(BASE_DIR)


# ---WARNING: INCOMPLETE---#
def sample(config="config.ini", save_file_path="carla_data/output"):
    """
    Sample from root directory using config, returns listfiles for train/test splits

    :param root: directory where data is located
    :type root: str
    :param config: configuration file specifying sampling parameters for train/test sets
    :type config: str
    :param save_file_path: directory where listfiles will be saved
    :type save_file_path: str
    """
    if save_file_path == "carla_data/output":
        save_file_path = os.path.join(os.getcwd(), BASE_DIR, save_file_path)

    conf = configparser.ConfigParser()
    conf.read(os.path.join(os.getcwd(), config))
    conf.sections()

    ALL = conf["General"]["ALL"]
    SPLITS = conf["General"]["SPLITS"]
    SAMPLE_SIZE = conf["General"]["SAMPLE_SIZE"]
    DATA_PATH = conf["Paths"]["DATA_PATH"]
    EGO_BEHAVIOR = [v.strip() for v in conf["Internal Variables"]["EGO_BEHAVIOR"].split(',')]
    EXTERNAL_BEHAVIOR = [v.strip() for v in conf["External Variables"]["EXTERNAL_BEHAVIOR"].split(',')]
    WEATHER = [int(v.strip()) for v in conf["External Variables"]["WEATHER"].split(',')]
    MAP = [v.strip() for v in conf["External Variables"]["MAP"].split(',')]

    DATA_PATH = os.path.join(os.getcwd(), BASE_DIR, DATA_PATH)
    SPLITS = literal_eval(SPLITS)
    # SPLITS = [float(x) for x in SPLITS]
    SAMPLE_SIZE = int(SAMPLE_SIZE)
    
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
        raise ValueError("Invalid configuration parameters. Please check config.ini.")

    frames = []
    os.chdir(DATA_PATH)
    for episode in os.listdir(DATA_PATH):
        if episode == ".gitignore":
            continue
        if episode == "calibmatrices.txt":
            continue
        os.chdir(episode)
        curr_dir = DATA_PATH + "/" + episode
        ep_conf = configparser.ConfigParser()
        ep_conf.read(os.path.join(curr_dir, config))
        ego_behav = ep_conf["Internal Variables"]["EGO_BEHAVIOR"]
        if not ALL and not (ego_behav in EGO_BEHAVIOR):
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
                list_dir = episode + "/" + iteration + "/" + timestamp

                it_conf = configparser.ConfigParser()
                it_conf.read(config)

                if ALL or (
                    it_conf["External Variables"]["EXTERNAL_BEHAVIOR"]
                    in EXTERNAL_BEHAVIOR
                    and it_conf["External Variables"]["WEATHER"] in WEATHER
                    and it_conf["External Variables"]["MAP"] in MAP
                ):
                    frames.extend(
                        [
                            frame
                            for frame in os.listdir(curr_dir)
                            if frame != "config.ini"
                        ]
                    )

    # randomly sample indices from directories
    assert len(frames) >= SAMPLE_SIZE
    random_indices = random.sample(range(len(frames)), SAMPLE_SIZE)
    random.shuffle(random_indices)

    train_len = int(SPLITS[0] * SAMPLE_SIZE)
    val_len = int(SPLITS[1] * SAMPLE_SIZE)

    train = random_indices[:train_len]
    val = random_indices[train_len : train_len + val_len]
    test = random_indices[train_len + val_len :]

    print(f"Wrote {len(train)} samples to train.csv")
    file = open(save_file_path + "/train.csv", "w")
    for idx in train:
        file.write(list_dir + "/" + frames[idx] + "\n")
    file.close()

    print(f"Wrote {len(val)} samples to val.csv")
    file = open(save_file_path + "/val.csv", "w")
    for idx in val:
        file.write(list_dir + "/" + frames[idx] + "\n")
    file.close()
    
    print(f"Wrote {len(test)} samples to test.csv")
    file = open(save_file_path + "/test.csv", "w")
    for idx in test:
        file.write(list_dir + "/" + frames[idx] + "\n")
    file.close()

    return save_file_path
