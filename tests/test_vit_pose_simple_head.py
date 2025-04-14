import unittest
import argparse
import torch

# Local imports
from util.utils import create_folder_if_missing
from model import TopdownHeatmapSimpleHead
from postprocess import _get_max_preds
from plotting import save_image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default="test_vit_pose_simple_head",
                        help="Result dir for files created by this test")
    parser.add_argument("--num_particles", type=int, default=17,
                        help="Number of particles that the model outputs as predictions")
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_heatmap_examples", type=int, default=5)

    args = parser.parse_args()

    return args


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()

        create_folder_if_missing(self.args.result_dir)

        self.model = TopdownHeatmapSimpleHead(in_channels=self.args.latent_dim,
                                              out_channels=self.args.num_particles)

    def test_output_shapes(self):
        image = torch.rand((1, 196, 768)).reshape((1, 768, 14, 14))

        output = self.model.forward(image)

        print(output.shape)
        print(torch.min(output))
        print(torch.max(output))

        keypoints = _get_max_preds(output.detach().cpu().numpy())
        print(keypoints)
        print(keypoints[0].shape)  # center coordinates
        print(keypoints[1].shape)  # confidence scores

    def test_heatmap_plotting(self):
        image = torch.rand((1, 196, 768)).reshape((1, 768, 14, 14))

        output = self.model.forward(image)

        for i in range(self.args.num_heatmap_examples):
            save_image(output[0, i:i+1], f"heatmap_example_{i}.png", self.args.result_dir)


if __name__ == '__main__':
    unittest.main()
