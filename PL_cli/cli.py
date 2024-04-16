import argparse
import sys

from edit_config import edit_config

def parse_args():
    parser = argparse.ArgumentParser(description="This is the command line interface for the Pseudo-LiDAR project.")

    parser.add_argument("--key", type=str, help="The key to modify. Options: data_path, ego_behavior, external_behavior, weather, map, all, splits, sample_size, carla_python_path, poll_rate, camera_x, camera_y, camera_fov")
    parser.add_argument("--value", type=str, help="The new value for the key.")
    parser.add_argument("--file-path", type=str, help="Path to the config file.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        edit_config(args.key, args.value, args.file_path)
        sys.exit(0)
    except Exception as e:
        print(e)
        print("Usage: python PL_cli [-h] [--key KEY] [--value VALUE] [--file-path FILE_PATH]")
        sys.exit(1)

if __name__ == "__main__":
    main()
