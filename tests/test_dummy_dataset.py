import unittest

from torch.utils.data import DataLoader, random_split

# Local imports
from dataset import DummyDataset

TEST_DATASET_PATH = '../dataset/dummy_dataset/data'
TRAIN_EVAL_SPLIT = 0.9
BATCH_SIZE = 8


class DummyDatasetTests(unittest.TestCase):
    def setUp(self):
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

        for micrographs, targets in train_dataloader:
            pass

        for micrographs, targets in test_dataloader:
            pass


if __name__ == '__main__':
    unittest.main()
