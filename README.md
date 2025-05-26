# Masters-Thesis-Marko-Janic
This repository contains all code used for my Master's thesis.

In this thesis we explore how powerful pretrained vision transformers can bring in the task of particle picking in cryo 
electron microscopy. The first task of this thesis is implement and analyze existing state-of-the-art deep learning 
methods. The second task of this thesis is to explore the interest of using pretrained vision transformers in a 
straightforward particle picking pipeline. This includes a thorough understanding of the cryoEM problem, 
its implementation using existing Python functions and the collection of meaningful real or synthetic dataset. The 
evaluation of the different approaches will be done by following the formalism described in the CryoSPARC paper, by 
reconstructing a single averaged particles from the estimation of the previous methods.

## Development
### Description of pipeline
Input of size batch_size x 3 x 224 x 224 -> vit model preprocessor -> vit model -> output of size batch_size x 197 x 768
-> Resize to batch_size x 768 x 14 x 14 -> model -> output of size batch_size x num_predictions x 112 x 112 -> 
calculate predictions by taking local maxima of each heatmap and checking if it's above the threshold we set.

The output of the vit model is 197 due to the embedded patches + class token. We don't use the class token.

### Important arguments
- one_heatmap: If set to true the targets for the model will contain one heatmap with multiple gaussians on it instead 
of having one heatmap per prediction.
- particle_width
- particle_height
- mode
- add_noise
- noise
- gaussians_3d: Will create a 3d volume with 3d gaussians around particle locations and then feed flat slices of that
volume to the model instead of making 2d target heatmaps with 2d gaussians.
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

## Relevant Papers:
- An Image is Worth 16x16 Words (I use their pretrained ViT): 
  - https://arxiv.org/abs/2010.11929
  - https://github.com/google-research/vision_transformer
- CryoTransformer (I use their loss function): 
  - https://academic.oup.com/bioinformatics/article/40/3/btae109/7614090
  - https://github.com/jianlin-cheng/CryoTransformer
