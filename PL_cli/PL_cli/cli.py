import argparse
import sys

from model import hello
from edit_config import edit_config

def parse_args():
    parser = argparse.ArgumentParser(description="This is the command line interface for the pseudo-lidar project.")
    # parser.add_argument("--hello", action="store_true", help="Print 'Hello!'")
    # parser.add_argument("--edit-config-processing", action="store_true", help="Edit the data processing parameters.")
    # parser.add_argument("--edit-sampling", action="store_true", help="Edit the sampling parameters.")
    # parser.add_argument("--run-sampling", action="store_true", help="Run the sampling process.")
    # parser.add_argument("--edit-model", action="store_true", help="Edit the model parameters.")
    
    parser.add_argument("--function", type=str, help="The function to run. Options: \nhello\nedit_config\n")
    parser.add_argument("--key", type=str, help="The key to modify.")
    parser.add_argument("--value", type=str, help="The new value for the key.")
    parser.add_argument("--file-path", type=str, help="Path to the config file.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    # print(args)
    if args.function == "hello":
        hello()
        sys.exit(0)

    if args.function == "edit_config":
        print("Editing configuration file...")
        edit_config(args.file_path, "../processing/dummy_config.ini", args.key, args.value)
        sys.exit(0)

    print("Usage: python PL_cli --function [function] --key [key] --value [value] --file-path [file_path]")
    sys.exit(1)

if __name__ == "__main__":
    main()
