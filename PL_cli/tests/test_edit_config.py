import unittest
import argparse
import sys
sys.path.insert(0, '..\PL_cli\\')
from PL_cli.edit_config import edit_config

class TestEditConfig(unittest.TestCase):
    def test_edit_config(self):
        # Test valid inputs
        edit_config("config.ini", "dummy_config.ini", "all", "True")
        edit_config("config.ini", "dummy_config.ini", "all", "False")
        edit_config("config.ini", "dummy_config.ini", "splits", "(0.1, 0.2, 0.7)")
        edit_config("config.ini", "dummy_config.ini", "sample_size", "100")
        edit_config("config.ini", "dummy_config.ini", "data_path", "data")
        edit_config("config.ini", "dummy_config.ini", "ego_behavior", "normal")
        edit_config("config.ini", "dummy_config.ini", "external_behavior", "aggressive")
        edit_config("config.ini", "dummy_config.ini", "weather", "1")
        edit_config("config.ini", "dummy_config.ini", "map", "Town01")

        # Test invalid inputs
        with self.assertRaises(ValueError, argparse.ArgumentTypeError):
            edit_config("config.ini", "dummy_config.ini", "all", "3")
            edit_config("config.ini", "dummy_config.ini", "splits", "(0.1, 0.2)")
            edit_config("config.ini", "dummy_config.ini", "sample_size", "0")
            edit_config("config.ini", "dummy_config.ini", "data_path", "nonexistent")
            edit_config("config.ini", "dummy_config.ini", "ego_behavior", "bad")
            edit_config("config.ini", "dummy_config.ini", "external_behavior", "bad")
            edit_config("config.ini", "dummy_config.ini", "weather", "13")
            edit_config("config.ini", "dummy_config.ini", "map", "bad")

if __name__ == "__main__":
    unittest.main()