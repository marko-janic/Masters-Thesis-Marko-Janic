# Masters-Thesis-Marko-Janic
This repository contains all code used for my Master's thesis.

## Development

### Normal
Create virtual environment:
```
python -m venv venv
```
Activate virtual environment (Windows):
```
.\venv\Scripts\activate
```
Install requirements:
```
pip install -r requirements.txt
```

### On SciCore
Login to SciCore:
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
Run script:
```
python main.py --config run_configs/default_dummy_dataset_training.json
```

### Recommended Folder structure
TODO: Add script for creating folder structure, describe folder structure

## Relevant Papers:
- Vision Transformer Model Google
    - https://arxiv.org/abs/2010.11929
    - Colab Notebook: https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb
- Pytorch implementation of ViTs:
  - Implementation: https://pytorch.org/vision/main/models/vision_transformer.html
  - Fine-tuning it?: https://github.com/jeonsworld/ViT-pytorch
