#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=finetuning_ablation_study_from_4_layers_up    #Name of your job
#SBATCH --time=7-00:00:00    #Maximum allocated time
#SBATCH --qos=gpu1week      #Selected queue to allocate your job (the gpu before 6hours is important, see comment: https://wiki.biozentrum.unibas.ch/display/scicore/9.+Requesting+GPUs)
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:1          #--gres=gpu:2 for two GPU, etc
#SBATCH --output=job_results/finetuning_ablation_study_from_4_layers_up.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=job_results/finetuning_ablation_study_from_4_layers_up.e%j    #Path and name to the file for the STDERR
#SBATCH --cpus-per-task=16     #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve

module purge
module load Miniconda3
source activate masters_thesis_marko_janic
conda list

#python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/finetune_1_layers.json
#python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/finetune_2_layers.json
python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/finetune_4_layers.json
python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/finetune_8_layers.json
python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/finetune_16_layers.json
python main.py --config run_configs/new_ablation_studies/finetuning_vit_10dB/train_vit_from_scratch.json
