import argparse
import sys

from model import hello

def parse_args():
    parser = argparse.ArgumentParser(description="This is the command line interface for the pseudo-lidar project.")
    parser.add_argument("--hello", action="store_true", help="Print 'Hello!'")
    parser.add_argument("--edit-config", action="store_true", help="Edit the configuration file.")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    hello()
    sys.exit(0)

if __name__ == "__main__":
    main()
