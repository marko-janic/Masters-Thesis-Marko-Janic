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

### Recommended Folder structure
TODO

### General
Create virtual environment:
```
python -m venv venv
```
Install requirements from requirements file:
```
pip install -r requirements.txt
```
Write new requirements to requirements file:
```
pip freeze > requirements.txt
```

### Windows
Activate virtual environment (Windows):
```
.\venv\Scripts\activate
```

### sciCORE (Linux)
Login to SciCore (will ask you for password):
```
ssh -Y <username>@login12.scicore.unibas.ch
```
Load python module:
```
module purge
module load Python/3.10.8-GCCcore-12.2.0
```
Activate environment:
```
source venv/bin/activate
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
