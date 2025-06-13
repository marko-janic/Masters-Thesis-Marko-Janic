# Cryo-Et particle picker with ViT backbone
This repository contains all code used for my Master's thesis. In this thesis I train a deconvolution model on the output of pretrained ViT backbones to perform particle picking in the context of Cryo-ET.

## Development
### Shrec Particles
- 4V94:     Not included
- 4CR2:     Increase by 2, 2, 3
- 1QVR:     Fine
- 1BXN:     Looks fine, Fits fine with 15 15 10
- 3CF3:     Looks like a flower, fine with 15 15 10
- 1U6G:     Looks fine with 15 15 10
- 3D2F:     Looks fine, particle is slightly smaller than 15 15 10
- 2CG9:     Looks fine, maybe slightly smaller than 15 15 10
- 3H84:     Looks fine and, like an "S", fits fine with 15 15 10
- 3GL1:     Looks fine, Slightly smaller than 15 15 10
- 1S3X:     Looks fine, a bit smaller than 15 15 10
- 5MRC:     Increase by 8, 8, 5
- vesicle:  Not included
- fiducial: Looks fine, some are slightly smaller thant 15 15 10

### Description of pipeline
Input of size batch_size x 3 x 224 x 224 -> vit model preprocessor -> vit model -> output of size batch_size x 197 x 768
-> Resize to batch_size x 768 x 14 x 14 -> model -> output of size batch_size x num_predictions x 112 x 112 -> 
calculate predictions by taking local maxima of each heatmap and checking if it's above the threshold we set.

The output of the vit model is 197 due to the embedded patches + class token. We don't use the class token.
The 768 is referred to as "latent dimension" and can be different depending on what kind of ViT model we use.

### Important arguments
- particle_width
- particle_height
- particle_depth
- mode
- add_noise
- noise
- use_fbp

### Adding own Dataset
Use Dataset class from torch. Make sure all images are 224 x 224. Make sure all images are between 0 and 1.

### Recommended Folder structure
TODO

## Installation
### General
Create conda environment from conda_environment.yml:
```
conda env create -f conda_environment.yml
```
Add new dependencies to conda environment if necessary (probably won't be):
```
conda env export > conda_environment.yml
```

### sciCORE (Linux)
Login to SciCore (will ask you for password):
```
ssh -Y <username>@login12.scicore.unibas.ch
```
Load python module:
```
module purge
module load Miniconda3
```
Activate environment:
```
conda activate masters_thesis_marko_janic
```
Run script (example):
```
python main.py --config run_configs/default_dummy_dataset_training.json
```
Run bash script in SLURM (example):
```
sbatch slurm_scripts/create_dummy_dataset.sh
```

### Linting
I try to use the default linter of PyCharm for this code wherever I don't forget :p

## Acknowledgement of sciCORE
Calculations were performed at sciCORE (http://scicore.unibas.ch/) scientific computing center at University of Basel.
Thank you very much!

## Link to thesis
TODO

## Citing
```
@mastersthesis{janic2025,
    author       = {Marko Janic},
    title        = {Vision Transformers for Cryo-ET particle picking},
    school       = {University of Basel},
    year         = {2025},
    month        = {July},
    note         = {Unpublished Master's thesis},
}
```
