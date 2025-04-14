import unittest
import argparse
import os

import matplotlib.pyplot as plt

# Local imports
from dataset import DummyDataset
from util.utils import create_folder_if_missing
from vit_model import get_vit_model, get_encoded_image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default="test_vit_model_huggingface",
                        help="Result dir for files created by this test")
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--dataset_path", type=str, default='../dataset/dummy_dataset_no_overlap/data')

    args = parser.parse_args()

    return args


class VitModelHuggingFaceTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()

        create_folder_if_missing(self.args.result_dir)

        self.vit_model, self.vit_image_processor = get_vit_model()
        self.dataset = DummyDataset(dataset_size=self.args.dataset_size, dataset_path=self.args.dataset_path)

    def test_shapes(self):
        micrograph, _ = self.dataset.__getitem__(0)
        micrograph = micrograph.unsqueeze(0)  # get it to batch x channel x h x w
        outputs = get_encoded_image(micrograph, self.vit_model, self.vit_image_processor)
        print(f"Output last hidden state shape: {outputs['last_hidden_state'].shape}")
        print(f"Output pooler_output shape: {outputs['pooler_output'].shape}")
        print(f"Output hidden_states shape: tuple of size {len(outputs['hidden_states'])}, each of shape "
              f"{outputs['hidden_states'][0].shape} ")

    def test_vit_image_processor(self):
        micrograph, _ = self.dataset.__getitem__(0)
        micrograph = micrograph.unsqueeze(0)  # get it to batch x channel x h x w
        processed_micrograph = self.vit_image_processor(images=micrograph, return_tensors='pt', do_rescale=False)
        print(f"Processed_micrograph shape: {processed_micrograph['pixel_values'].shape}")

        plt.imshow(processed_micrograph['pixel_values'][0].permute(1, 2, 0).cpu().numpy())
        plt.savefig(os.path.join(self.args.result_dir, "processed_micrograph"))


if __name__ == '__main__':
    unittest.main()
