import argparse
from configparser import ConfigParser
import os

def edit_config(key: str, value: str):
    configs = ConfigParser(comment_prefixes="#", allow_no_value=True)
    configs.read('processing/config.ini')

    # input validation for "all" key
    if key == "all":
        true_values = ["True", "true", "TRUE", "t", "T", "1"]
        false_values = ["False", "false", "FALSE", "f", "F", "0"]
        if value in true_values:
            configs["General"]["all"] = str(True)
        elif value in false_values:
            configs["General"]["all"] = str(False)
        else:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be a boolean.")

    # input validation for "splits" key
    elif key == "splits":
        try:
            # Parse input string to tuple of floats
            values = tuple(map(float, value.strip('()').split(',')))
            # Check if tuple has three values
            if len(values) != 3:
                raise ValueError(f"Error: Tuple {values} must have three values.")
            # Check if values are between 0 and 1
            for v in values:
                if v < 0 or v > 1:
                    raise ValueError(f"Error: Value {v} must be between 0 and 1.")
            # Check if values sum to 1 within 1e-6
            if sum(values) - 1 > 1e-6:
                raise ValueError(f"Error: Values {values} must sum to 1.")
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
        
        configs["General"]["splits"] = str(values)

    # input validation for "sample size" key 
    elif key == "sample_size":
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be an integer.")
        if value < 10:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be at least 10.")
        # TODO: Add a check for the maximum value of sample_size
        # TODO: Add a way to use all frames

        configs["General"]["sample_size"] = str(value)

    # input validation for "data_path" key
    elif key == "data_path":
        if not os.path.exists(value):
            raise argparse.ArgumentTypeError(f"Error: Path to {value} not found.")
        
        configs["Path"]["data_path"] = value

    # input validation for ego_behavior
    elif key == "ego_behavior":
        value = value.lower()
        if value not in ["normal", "aggressive", "cautious"]:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be 'normal', 'aggressive', or 'cautious'.")

        configs["Internal Variables"]["ego_behavior"] = value

    # input validation for external_behavior
    elif key == "external_behavior":
        value = value.lower()
        if value not in ["normal", "aggressive", "cautious"]:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be 'normal', 'aggressive', or 'cautious'.")

        configs["External Variables"]["external_behavior"] = value

    # input validation for "weather" key
    elif key == "weather":
        # must be an integer
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be an integer.")
        # must be 1, 2, 5, 8, 9, or 12
        if int(value) not in [1, 2, 5, 8, 9, 12]:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be 1, 2, 5, 8, 9, or 12.")
        
        configs["External Variables"]["weather"] = str(value)

    # input validation for "map" key
    elif key == "map":
        value = value.capitalize()
        # must be "Town01", "Town02", "Town07"
        if value not in ["Town01", "Town02", "Town07"]:
            raise argparse.ArgumentTypeError(f"Error: Value {value} must be 'Town01', 'Town02', or 'Town07'.")

        configs["External Variables"]["map"] = value

    else:
        raise argparse.ArgumentTypeError(f"Error: Key {key} not found.")

    # write the new config file
    out_file_path = "./processing/config.ini"
    with open(out_file_path, "w") as f:
        configs.write(f)

    print(f"Key {key} updated to {value} in {out_file_path}.")
    return

