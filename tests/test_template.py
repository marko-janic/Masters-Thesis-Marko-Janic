"""
This file is only so you can easily create new tests if necessary
"""
import unittest
import argparse

# Local imports
from util.utils import create_folder_if_missing


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default="test_template",
                        help="Result dir for files created by this test")

    args = parser.parse_args()

    return args


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()

        create_folder_if_missing(self.args.result_dir)

    def test1(self):
        pass

    def test2(self):
        pass


if __name__ == '__main__':
    unittest.main()
