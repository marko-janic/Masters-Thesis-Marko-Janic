#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=vitpose_target_heatmaps_fixed_bigger_target_heatmaps     #Name of your job
#SBATCH --time=7-00:00:00    #Maximum allocated time
#SBATCH --qos=gpu1week       #Selected queue to allocate your job (the gpu before 6hours is important, see comment: https://wiki.biozentrum.unibas.ch/display/scicore/9.+Requesting+GPUs)
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:1          #--gres=gpu:2 for two GPU, etc
#SBATCH --output=job_results/vitpose_target_heatmaps_fixed_bigger_target_heatmaps.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=job_results/vitpose_target_heatmaps_fixed_bigger_target_heatmaps.e%j    #Path and name to the file for the STDERR
#SBATCH --cpus-per-task=16     #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve

module purge
module load Miniconda3
conda activate masters_thesis_marko_janic

python main.py --config run_configs/dummy_dataset_training.json
