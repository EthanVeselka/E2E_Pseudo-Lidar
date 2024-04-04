import unittest
import argparse
import sys
sys.path.insert(0, '..\PL_cli\\')
sys.path.insert(0, '..\processing\\')
from PL_cli.edit_config import edit_config

class TestEditConfig(unittest.TestCase):
    def test_edit_config_valid_inputs(self):
        edit_config("all", "True")

    def test_edit_config_valid_inputs2(self):
        edit_config("all", "False")

    def test_edit_config_valid_inputs3(self):
        edit_config("splits", "(0.1, 0.2, 0.7)")

    def test_edit_config_valid_inputs4(self):
        edit_config("sample_size", "100")

    def test_edit_config_valid_inputs5(self):
        edit_config("map", "Town01")

    def test_edit_config_valid_inputs6(self):
        edit_config("ego_behavior", "normal")

    def test_edit_config_valid_inputs7(self):
        edit_config("external_behavior", "aggressive")

    def test_edit_config_valid_inputs8(self):
        edit_config("weather", "1")



    def test_edit_config_invalid_inputs(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("all", "3")

    def test_edit_config_invalid_inputs2(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("splits", "(0.1, 0.2)")

    def test_edit_config_invalid_inputs3(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("sample_size", "0")

    def test_edit_config_invalid_inputs4(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("data_path", "nonexistent")

    def test_edit_config_invalid_inputs5(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("ego_behavior", "bad")

    def test_edit_config_invalid_inputs6(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("external_behavior", "bad")

    def test_edit_config_invalid_inputs7(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("weather", "13")

    def test_edit_config_invalid_inputs8(self):
        # Test invalid input
        with self.assertRaises(Exception):
            edit_config("map", "bad")

if __name__ == "__main__":
    unittest.main()