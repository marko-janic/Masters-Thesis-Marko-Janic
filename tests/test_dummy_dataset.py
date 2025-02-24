import unittest
import os

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

# Local imports
from dataset import DummyDataset
from util.utils import create_folder_if_missing

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'
TEST_RESULTS_FOLDER = 'test_dummy_dataset'
TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8
EXAMPLE_VISUALIZATIONS = 3


class DummyDatasetTests(unittest.TestCase):
    def setUp(self):
        create_folder_if_missing(TEST_RESULTS_FOLDER)
        self.dataset = DummyDataset(dataset_path=TEST_DATASET_PATH)

    def test_shapes(self):
        pass

    def test_micrographs_normalization(self):
        for i in range(self.dataset.__len__()):
            micrograph, target = self.dataset.__getitem__(i)
            self.assertGreaterEqual(micrograph.min(), 0)
            self.assertGreaterEqual(target['boxes'].min(), 0)
            self.assertLessEqual(micrograph.max(), 1)
            self.assertLessEqual(target['boxes'].max(), 1)

    def test_dataset_dataloader(self):
        train_size = int(TRAIN_EVAL_SPLIT * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        for micrographs, target_indexes in train_dataloader:
            targets = []
            for target_index in target_indexes:
                targets.append(self.dataset.targets[target_index])
            pass

        for micrographs, targets in test_dataloader:
            pass

    def test_dataset_images(self):
        for i in range(EXAMPLE_VISUALIZATIONS):
            micrograph, target = self.dataset.__getitem__(i)
            plt.imshow(micrograph.permute(1, 2, 0))
            plt.savefig(os.path.join(TEST_RESULTS_FOLDER, f'example_{i}.png'))


if __name__ == '__main__':
    unittest.main()
