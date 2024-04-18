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

    config = "config.ini"
    save_file_path = "carla_data/output"
    datapath = "carla_data/example_data"

    args = parser.parse_args()
    if args.clean:
        utils.clean_output(datapath)
    elif args.gen_disp:
        gd.generate_disparity(datapath)
    elif args.sample:
        sample.sample(config, save_file_path)
