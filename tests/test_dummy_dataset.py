import unittest

# Local imports
from dataset import DummyDataset

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'


class ShrecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDataset()

    def test_shapes(self):
        pass

    def test_micrographs_normalization(self):
        pass


if __name__ == '__main__':
    unittest.main()
