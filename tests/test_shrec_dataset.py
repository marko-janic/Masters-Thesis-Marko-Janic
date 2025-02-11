import unittest

# Local imports
from dataset import ShrecDataset


TEST_DATASET_SAMPLING_POINTS = 16
TEST_DATASET_PATH = '../dataset/shrec21_full_dataset/'


class ShrecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ShrecDataset(TEST_DATASET_SAMPLING_POINTS, dataset_path=TEST_DATASET_PATH)

    def test_shapes(self):
        sub_micrograph, coordinates = self.dataset.__getitem__(0)
        micrograph_shape = self.dataset.micrograph.shape
        dataset_length = self.dataset.__len__()

        self.assertEqual(micrograph_shape, (512, 512))
        self.assertEqual(sub_micrograph.shape, (3, 224, 224))
        self.assertEqual((coordinates[0], coordinates[1]), (0, 0))
        self.assertEqual(dataset_length, TEST_DATASET_SAMPLING_POINTS**2)


if __name__ == '__main__':
    unittest.main()
