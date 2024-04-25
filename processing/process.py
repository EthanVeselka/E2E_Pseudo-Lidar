import argparse
import sample 
import utils
from processing.pseudo_lidar import generate_disp as gd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean", action="store_true", help="Clean frame output folders"
    )
    parser.add_argument(
        "--gen_disp", action="store_true", help="Generate GT disparities for all frames"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Generate train/val/test splits"
    )
    parser.add_argument(
        "--image", action="store_true", help="Generate disp images"
    )

    config = "config.ini"
    save_file_path = "carla_data/output"
    datapath = "carla_data/data"

    args = parser.parse_args()
    if args.clean:
        utils.clean_output(datapath)
    elif args.gen_disp:
        gd.generate_disparity(datapath, args.image)
    elif args.sample:
        sample.sample(config, save_file_path)
