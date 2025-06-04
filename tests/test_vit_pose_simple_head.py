import unittest
import argparse
import torch

from torch.utils.data import DataLoader

# Local imports
from utils import create_folder_if_missing
from model import TopdownHeatmapSimpleHead
from plotting import save_image, compare_heatmaps_with_ground_truth
from dataset import DummyDataset
from train import create_heatmaps_from_targets

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default="test_vit_pose_simple_head",
                        help="Result dir for files created by this test")
    parser.add_argument("--num_particles", type=int, default=7,
                        help="Number of particles that the model outputs as predictions")
    parser.add_argument("--latent_dim", type=int, default=768, help="Dimensions of input to model")
    parser.add_argument("--num_heatmap_examples", type=int, default=5)
    parser.add_argument("--num_heatmap_comparison_examples_batches", type=int, default=2)

    parser.add_argument("--particle_width", type=int, default=80)
    parser.add_argument("--particle_height", type=int, default=80)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--dataset_path", type=str, default="../dataset/dummy_dataset_no_overlap/data")
    parser.add_argument("--batch_size", type=int, default=4, help="Size of each training batch")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    return args


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()

        create_folder_if_missing(self.args.result_dir)

        self.model = TopdownHeatmapSimpleHead(in_channels=self.args.latent_dim,
                                              out_channels=self.args.num_particles)

        self.dataset = DummyDataset(dataset_size=self.args.dataset_size, dataset_path=self.args.dataset_path,
                               particle_width=self.args.particle_width, particle_height=self.args.particle_height)
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, drop_last=True)

    def test_output_shapes(self):
        image = torch.rand((1, 196, 768)).reshape((1, 768, 14, 14))

        output = self.model.forward(image)
        heatmaps = output["heatmaps"]
        logits = output["pred_logits"]
        keypoints = output["pred_boxes"]

        print(heatmaps.shape)
        print(torch.min(heatmaps))
        print(torch.max(heatmaps))
        print(logits)
        print(keypoints)

    def test_heatmap_plotting(self):
        image = torch.rand((1, 196, 768)).reshape((1, 768, 14, 14))

        output = self.model.forward(image)
        heatmaps = output["heatmaps"]

        for i in range(self.args.num_heatmap_examples):
            save_image(heatmaps[0, i:i+1], f"heatmap_example_{i}.png", self.args.result_dir)

    def test_generate_heatmaps_from_targets(self):
        for batch_index, (micrograph, index) in enumerate(self.dataloader):
            if batch_index >= self.args.num_heatmap_comparison_examples_batches:
                break

            targets = self.dataset.get_targets_from_target_indexes(index, self.args.device)
            target_heatmaps = create_heatmaps_from_targets(targets, num_predictions=self.args.num_particles,
                                                           device=self.args.device)

            for i in range(len(target_heatmaps)):
                compare_heatmaps_with_ground_truth(micrograph=micrograph[i],
                                                   particle_locations=targets[i]["boxes"],
                                                   heatmaps=target_heatmaps[i],
                                                   heatmaps_title="Generated Heatmaps from targets",
                                                   result_folder_name=f"ground_truth_vs_generated_heatmaps_batch_"
                                                                      f"{batch_index}_example_{i}",
                                                   result_dir=self.args.result_dir)


if __name__ == '__main__':
    unittest.main()
