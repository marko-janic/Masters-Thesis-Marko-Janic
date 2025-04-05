import unittest

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates


TEST_DATASET_SAMPLING_POINTS = 16
TEST_DATASET_PATH = '../dataset/shrec21_full_dataset/'


class ShrecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ShrecDataset(TEST_DATASET_SAMPLING_POINTS, dataset_path=TEST_DATASET_PATH)

    def test_shapes(self):
        sub_micrograph, coordinate_tl = self.dataset.__getitem__(0)
        micrograph_shape = self.dataset.micrograph.shape
        dataset_length = self.dataset.__len__()

        self.assertEqual(micrograph_shape, (512, 512))
        self.assertEqual(sub_micrograph.shape, (3, 224, 224))
        self.assertEqual((coordinate_tl[0], coordinate_tl[1]), (0, 0))
        self.assertEqual(dataset_length, TEST_DATASET_SAMPLING_POINTS**2)

    def test_micrographs_normalization(self):
        sub_micrograph, coordinate_tl = self.dataset.__getitem__(0)

        # Micrograph should be normalized between 0 and 1
        self.assertEqual(self.dataset.micrograph.min(), 0)
        self.assertEqual(self.dataset.micrograph.max(), 1)
        self.assertGreaterEqual(sub_micrograph.min(), 0)
        self.assertLessEqual(sub_micrograph.max(), 1)

    def test_get_particle_locations(self):
        sub_micrograph, coordinate_tl = self.dataset.__getitem__(0)
        particle_locations = get_particle_locations_from_coordinates(coordinate_tl, self.dataset.sub_micrograph_size,
                                                                     self.dataset.particle_locations)


if __name__ == '__main__':
    unittest.main()
