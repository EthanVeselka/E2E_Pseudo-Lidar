from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import time
import math
import sys
from PIL import Image


BASE_DIR = "../../.."
sys.path.append(BASE_DIR)

from models.disparity_model.stackhourglass import PSMNet as stackhourglass
from logger import setup_logger
from processing.pseudo_lidar import transforms as preprocess
import processing.pseudo_lidar.custom_loader as ls
import processing.pseudo_lidar.custom_dataset as DA

parser = argparse.ArgumentParser(description="PSMNet")
parser.add_argument("--loadmodel", default=None, help="loading model")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument(
    "--all", action="store_true", help="predict all frames or just test"
)
parser.add_argument(
    "--cuda", action="store_true", default=False, help="Enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--datapath",
    default="carla_data/example_data/",
    help="datapath",
)
parser.add_argument(
    "--split_file",
    default="carla_data/output",
    help="training data sampling indices",
)
parser.add_argument(
    "--save_figure",
    action="store_true",
    help="if true save the png file",
)
args = parser.parse_args()


args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


datapath = os.path.join(BASE_DIR, args.datapath)
split_file = os.path.join(BASE_DIR, args.split_file)
task = "all" if args.all else "test"
save_path = "../predictions" if (task == "test") else "output"
test_left_img, test_right_img, true_disps = ls.dataloader(datapath, split_file, task)

model = stackhourglass(args.maxdisp)

if args.cuda:
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict["state_dict"])

print(
    "Number of model parameters: {}".format(
        sum([p.data.nelement() for p in model.parameters()])
    )
)


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()
    else:
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)

    with torch.no_grad():
        print(imgL.size())
        print(imgR.size())
        output = model(imgL, imgR)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    if task == "test" and not os.path.exists(save_path):
        os.mkdir(save_path)

    for idx in range(len(test_left_img)):
        imgL_o = Image.open(test_left_img[idx]).convert("RGB")
        imgR_o = Image.open(test_right_img[idx]).convert("RGB")

        w = 1248
        h = 384
        imgL = np.array(imgL_o.crop((0, 0, w, h))).astype("float32")
        imgR = np.array(imgR_o.crop((0, 0, w, h))).astype("float32")
        # imgL = np.array(imgL).astype("float32")
        # imgR = np.array(imgR).astype("float32")
        imgL = processed(imgL).numpy()
        imgR = processed(imgR).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        top_pad = 384 - imgL.shape[2]
        left_pad = 1248 - imgL.shape[3]
        imgL = np.lib.pad(
            imgL,
            ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
            mode="constant",
            constant_values=0,
        )
        imgR = np.lib.pad(
            imgR,
            ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
            mode="constant",
            constant_values=0,
        )

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        print("time = %.2f" % (time.time() - start_time))

        # top_pad = 384 - 352
        # left_pad = 1248 - 1200
        # img = pred_disp[top_pad:, :-left_pad]
        img = pred_disp
        print(test_left_img[idx].split("/")[-1])
        frame = test_left_img[idx].split("/")[-2]
        if args.save_figure:
            if task == "test":
                Image.fromarray(img).convert("RGB").save(
                    save_path + f"/predicted_disp_{frame}.png",
                )
            else:
                Image.fromarray(img).convert("RGB").save(
                    "/".join(test_left_img[idx].split("/")[:-1])
                    + "/"
                    + save_path
                    + "/predicted_disp.png",
                )
        else:
            if task == "test":
                np.save(save_path + f"/predicted_disp_{frame}.npy", img)
            else:
                np.save(
                    "/".join(test_left_img[idx].split("/")[:-1])
                    + "/"
                    + save_path
                    + "/predicted_disp.npy",
                    img,
                )


if __name__ == "__main__":
    main()
