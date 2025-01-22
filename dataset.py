import os

import mrcfile as mrc

from torch.utils.data import Dataset


def create_sub_micrographs(image, crop_size):
    height, width = image.shape
    num_crops_y = height // crop_size
    num_crops_x = width // crop_size

    assert num_crops_x > 0 and num_crops_y > 0, \
        "Image is smaller than crop size."

    image = image[:num_crops_y * crop_size, :num_crops_x * crop_size]

    crops = (
        image
        .reshape(num_crops_y, crop_size, num_crops_x, crop_size)  # Create blocks
        .transpose(0, 2, 1, 3)  # Reorder dimensions to (blocks_y, blocks_x, crop_size, crop_size)
        .reshape(-1, crop_size, crop_size)  # Flatten into list of crops
    )
    return crops


class ShrecDataset(Dataset):
    num_models = 10  # See shrec dataset
    projection_number = 29  # Which projection to use out of the 61 available. See alignment_simulated.txt files
    vit_input_size = 224  # The size we want our micrographs to be
    micrograph_size = 1024  # See shrec dataset
    num_crops_side = micrograph_size // vit_input_size  # We divide the micrograph into "sub-micrographs"
    num_sub_micrographs = num_crops_side * num_crops_side

    def __init__(self):
        self.data = []

        for i in range(self.num_models):
            with mrc.open(f'dataset/shrec21_full_dataset/model_{i}/projections_unbinned.mrc',
                          permissive=True) as f:
                micrograph = f.data[29]
                sub_micrographs = create_sub_micrographs(micrograph, self.vit_input_size)

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None
