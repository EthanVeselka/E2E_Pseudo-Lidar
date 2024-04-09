import argparse
import sys

from edit_config import edit_config

def parse_args():
    parser = argparse.ArgumentParser(description="This is the command line interface for the Pseudo-LiDAR project.")

    # parser.add_argument("--function", type=str, help="The function to run. Options: \nhello\nedit_config\n")
    parser.add_argument("--key", type=str, help="The key to modify. Options: all, splits, sample_size, map, ego_behavior, external_behavior, weather")
    parser.add_argument("--value", type=str, help="The new value for the key.")
    # parser.add_argument("--file-path", type=str, help="Path to the config file.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    # print(args)
    # if args.function == "edit_config":
    #     print("Editing configuration file...")
    #     edit_config(args.key, args.value)
    #     sys.exit(0)

    try:
        edit_config(args.key, args.value)
        sys.exit(0)
    except Exception as e:
        print(e)
        print("Usage: python PL_cli --key [key] --value [value]")
        sys.exit(1)


if __name__ == "__main__":
    main()
