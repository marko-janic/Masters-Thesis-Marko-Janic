import unittest
import argparse
import os
import torch
import napari

import matplotlib.pyplot as plt
import mrcfile as mrc
import numpy as np

from skimage.feature import peak_local_max
from tqdm import tqdm

# Local imports
from dataset import ShrecDataset, get_particle_locations_from_coordinates
from plotting import save_image_with_bounding_object
from utils import create_folder_if_missing


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", type=str, default='test_shrec_dataset')
    parser.add_argument("--dataset_path", type=str, default='../dataset/shrec21_full_dataset/')
    parser.add_argument("--sampling_points", type=int, default=4)
    parser.add_argument("--z_slice_size", type=int, default=1)
    parser.add_argument("--min_z", type=int, default=170)
    parser.add_argument("--max_z", type=int, default=171)
    parser.add_argument("--example_visualizations", type=int, default=20)
    parser.add_argument("--model_number", type=int, default=[0])
    parser.add_argument("--particle_width", type=int, default=12)
    parser.add_argument("--particle_height", type=int, default=12)
    parser.add_argument("--particle_depth", type=int, default=6)
    parser.add_argument("--noise", type=int, default=10)
    parser.add_argument("--heatmap_size", type=int, default=112)
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--use_fbp", type=bool, default=True)
    parser.add_argument("--fbp_num_projections", type=int, default=60)
    parser.add_argument("--fbp_min_angle", type=int, default=-torch.pi/3)
    parser.add_argument("--fbp_max_angle", type=int, default=torch.pi/3)
    parser.add_argument("--shrec_specific_particle", type=str, default="")
    parser.add_argument("--use_shrec_reconstruction", type=bool, default=False)

    args = parser.parse_args()

    return args


class ShrecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.args = get_args()
        create_folder_if_missing(self.args.result_dir)
        self.dataset = ShrecDataset(sampling_points=self.args.sampling_points, dataset_path=self.args.dataset_path,
                                    z_slice_size=self.args.z_slice_size, particle_width=self.args.particle_width,
                                    particle_height=self.args.particle_height, noise=self.args.noise,
                                    add_noise=self.args.add_noise, min_z=self.args.min_z, max_z=self.args.max_z,
                                    use_fbp=self.args.use_fbp, fbp_num_projections=self.args.fbp_num_projections,
                                    fbp_min_angle=self.args.fbp_min_angle, fbp_max_angle=self.args.fbp_max_angle,
                                    particle_depth=self.args.particle_depth,
                                    shrec_specific_particle=self.args.shrec_specific_particle,
                                    model_number=self.args.model_number, heatmap_size=self.args.heatmap_size,
                                    use_shrec_reconstruction=self.args.use_shrec_reconstruction)

    def test_orientations(self):
        micrograph_0, heatmap_0, targets_0 = self.dataset.__getitem__(0)
        micrograph_1, heatmap_1, targets_1 = self.dataset.__getitem__(1)
        micrograph_2, heatmap_2, targets_2 = self.dataset.__getitem__(2)
        micrograph_3, heatmap_3, targets_3 = self.dataset.__getitem__(3)
        
        plt.imshow(micrograph_0.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Orientation 0: normal")
        plt.savefig(os.path.join(self.args.result_dir, "micrograph_orientation_0.png"))
        plt.close()
        plt.imshow(micrograph_1.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Orientation 1: horizontal flip")
        plt.savefig(os.path.join(self.args.result_dir, "micrograph_orientation_1.png"))
        plt.close()
        plt.imshow(micrograph_2.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Orientation 2: vertical flip")
        plt.savefig(os.path.join(self.args.result_dir, "micrograph_orientation_2.png"))
        plt.close()
        plt.imshow(micrograph_3.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Orientation 3: transposed")
        plt.savefig(os.path.join(self.args.result_dir, "micrograph_orientation_3.png"))
        plt.close()

        plt.imshow(heatmap_0[0].cpu().numpy())
        plt.title(f"Orientation 0: normal")
        plt.savefig(os.path.join(self.args.result_dir, "heatmap_orientation_0.png"))
        plt.close()
        plt.imshow(heatmap_1[0].cpu().numpy())
        plt.title(f"Orientation 1: horizontal flip")
        plt.savefig(os.path.join(self.args.result_dir, "heatmap_orientation_1.png"))
        plt.close()
        plt.imshow(heatmap_2[0].cpu().numpy())
        plt.title(f"Orientation 2: vertical flip")
        plt.savefig(os.path.join(self.args.result_dir, "heatmap_orientation_2.png"))
        plt.close()
        plt.imshow(heatmap_3[0].cpu().numpy())
        plt.title(f"Orientation 3: transposed")
        plt.savefig(os.path.join(self.args.result_dir, "heatmap_orientation_3.png"))
        plt.close()

        save_image_with_bounding_object(micrograph_0, targets_0*224, "box",
                                        {"box_width": 20, "box_height": 20},
                                        result_dir=self.args.result_dir,
                                        file_name=f"micrograph_{0}_with_bounding_boxes")
        save_image_with_bounding_object(micrograph_1, targets_1*224, "box",
                                        {"box_width": 20, "box_height": 20},
                                        result_dir=self.args.result_dir,
                                        file_name=f"micrograph_{1}_with_bounding_boxes")
        save_image_with_bounding_object(micrograph_2, targets_2*224, "box",
                                        {"box_width": 20, "box_height": 20},
                                        result_dir=self.args.result_dir,
                                        file_name=f"micrograph_{2}_with_bounding_boxes")
        save_image_with_bounding_object(micrograph_3, targets_3*224, "box",
                                        {"box_width": 20, "box_height": 20},
                                        result_dir=self.args.result_dir,
                                        file_name=f"micrograph_{3}_with_bounding_boxes")

    def test_3d_volume_local_maxima(self):
        heatmaps_volume = self.dataset.heatmaps_volume[self.args.model_number[0]].cpu().numpy()
        coordinates = torch.from_numpy(peak_local_max(heatmaps_volume, min_distance=6, threshold_abs=0.9))
        viewer = napari.Viewer()
        viewer.add_image(heatmaps_volume, name='Heatmaps Volume', colormap='magenta')
        viewer.add_points(coordinates, size=10, face_color='red')
        napari.run()

    def test_volume(self):
        heatmaps = self.dataset.heatmaps_volume[self.args.model_number[0]].cpu().numpy()

        viewer = napari.Viewer()

        for model_num in self.args.model_number:
            grandmodel = self.dataset.grandmodel[model_num].cpu().numpy()
            viewer.add_image(grandmodel, name=f'{model_num}', colormap='gray')

        #for layer in viewer.layers:
        #    if layer.__class__.__name__ == "Image":
        #        layer.contrast_limits = [0, 1]

        viewer.add_image(heatmaps, name='Heatmaps Volume', colormap='magenta', blending='additive')
        if self.dataset.use_fbp:
            fbp_volume = self.dataset.grandmodel_fbp[0].cpu().numpy()
            viewer.add_image(fbp_volume, name='FBP Reconstructed voume', colormap='gray', blending='additive')
        napari.run()

    def test_shapes(self):
        # TODO: fix these tests, they don't work right now
        sub_micrograph, coordinate_tl, _, _ = self.dataset.__getitem__(0)
        micrograph_shape = self.dataset.micrographs[0].shape
        dataset_length = self.dataset.__len__()

        self.assertEqual(micrograph_shape, (512, 512))
        self.assertEqual(sub_micrograph.shape, (3, 224, 224))
        self.assertEqual((coordinate_tl[0], coordinate_tl[1]), (0, 0))
        self.assertEqual(dataset_length, (self.args.sampling_points**2) * (self.dataset.grandmodel[0].shape[0] // self.dataset.z_slice_size))

    def test_backprojections_visualizations(self):
        with mrc.open(os.path.join(self.args.dataset_path, f"model_{self.args.model_number[0]}", "reconstruction.mrc"),
                      permissive=True) as f:
            print(f.data.shape)

    def test_visualize_z_slices(self):
        folder = os.path.join(self.args.result_dir, "z_slices")
        create_folder_if_missing(folder)

        for i in tqdm(range(len(self.dataset.micrographs)), desc="Visualizing z slices"):
            plt.imshow(self.dataset.micrographs[i][0])
            plt.title(f"Min z: {self.dataset.micrographs[i][1]}, Max: z: {self.dataset.micrographs[i][2]}")
            plt.savefig(os.path.join(folder, f"micrograph_slice_{i}.png"))
            plt.close()

    def test_example_visualizations_with_particles(self):
        # TODO: fix this test it doesn't work right now
        folder = os.path.join(self.args.result_dir, "dataset_examples_with_particles")
        create_folder_if_missing(folder)

        print("Dataset length: ", self.dataset.__len__())

        for i in range(15, 25):
            sub_micrograph, coordinate_tl, _, _ = self.dataset.__getitem__(i)
            selected_particles = get_particle_locations_from_coordinates(coordinates_tl=coordinate_tl,
                                                                         sub_micrograph_size=self.dataset.sub_micrograph_size,
                                                                         particle_locations=self.dataset.particle_locations,
                                                                         z_slice_size=self.dataset.z_slice_size,
                                                                         particle_width=20,
                                                                         particle_height=20)
            selected_particles["boxes"][:, 1] = self.dataset.sub_micrograph_size - selected_particles["boxes"][:, 1]

            save_image_with_bounding_object(image_tensor=sub_micrograph[0:1],
                                            particle_locations=selected_particles["boxes"],
                                            object_type="box", object_parameters={"box_width": 30, "box_height": 30},
                                            result_dir=folder, file_name=f"example_{i}.png")


if __name__ == '__main__':
    unittest.main()
