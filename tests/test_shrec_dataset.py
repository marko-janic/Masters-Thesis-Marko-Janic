import unittest
import argparse
import os
import torch

import matplotlib.pyplot as plt

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from plotting import save_image_with_bounding_object
from util.utils import create_folder_if_missing


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default='test_shrec_dataset')
    parser.add_argument("--dataset_path", type=str, default='../dataset/shrec21_full_dataset/')
    parser.add_argument("--sampling_points", type=int, default=4)
    parser.add_argument("--z_slice_size", type=int, default=20)
    parser.add_argument("--example_visualizations", type=int, default=20)

    args = parser.parse_args()

    return args


class ShrecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()
        create_folder_if_missing(self.args.result_dir)
        self.dataset = ShrecDataset(sampling_points=self.args.sampling_points, dataset_path=self.args.dataset_path,
                                    z_slice_size=self.args.z_slice_size)

    def test_shapes(self):
        sub_micrograph, coordinate_tl = self.dataset.__getitem__(0)
        micrograph_shape = self.dataset.micrographs[0].shape
        dataset_length = self.dataset.__len__()

        self.assertEqual(micrograph_shape, (512, 512))
        self.assertEqual(sub_micrograph.shape, (3, 224, 224))
        self.assertEqual((coordinate_tl[0], coordinate_tl[1]), (0, 0))
        self.assertEqual(dataset_length, (self.args.sampling_points**2) * (self.dataset.grandmodel.shape[0] // self.dataset.z_slice_size))

    def test_get_particle_locations(self):
        sub_micrograph, coordinate_tl = self.dataset.__getitem__(0)
        particle_locations = get_particle_locations_from_coordinates(coordinate_tl, self.dataset.sub_micrograph_size,
                                                                     self.dataset.particle_locations,
                                                                     z_slice_size=self.dataset.z_slice_size)

    def test_visualize_z_slices(self):
        folder = os.path.join(self.args.result_dir, "z_slices")
        create_folder_if_missing(folder)

        for i in range(len(self.dataset.micrographs)):
            plt.imshow(self.dataset.micrographs[i][0])
            plt.title(f"Min z: {self.dataset.micrographs[i][1]}")
            plt.savefig(os.path.join(folder, f"micrograph_slice_{i}.png"))
            plt.close()

    def test_example_visualizations_with_particles(self):
        folder = os.path.join(self.args.result_dir, "dataset_examples_with_particles")
        create_folder_if_missing(folder)

        print("Dataset length: ", self.dataset.__len__())

        for i in range(15, 25):
            sub_micrograph, coordinate_tl = self.dataset.__getitem__(i)
            selected_particles = get_particle_locations_from_coordinates(coordinates_tl=coordinate_tl,
                                                                         sub_micrograph_size=self.dataset.sub_micrograph_size,
                                                                         particle_locations=self.dataset.particle_locations,
                                                                         z_slice_size=self.dataset.z_slice_size)
            selected_dropped_columns = selected_particles[['X', 'Y']].to_numpy()
            selected_particles_tensor = torch.tensor(selected_dropped_columns)
            selected_particles_tensor[:, 1] = self.dataset.sub_micrograph_size - selected_particles_tensor[:, 1]

            save_image_with_bounding_object(image_tensor=sub_micrograph[0:1],
                                            particle_locations=selected_particles_tensor,
                                            object_type="box", object_parameters={"box_width": 30, "box_height": 30},
                                            result_dir=folder, file_name=f"example_{i}.png")


if __name__ == '__main__':
    unittest.main()
