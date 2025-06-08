#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=particle_1U6G_fbp_noise_20db     #Name of your job
#SBATCH --time=7-00:00:00    #Maximum allocated time
#SBATCH --qos=gpu1week       #Selected queue to allocate your job (the gpu before 6hours is important, see comment: https://wiki.biozentrum.unibas.ch/display/scicore/9.+Requesting+GPUs)
#SBATCH --partition=titan
#SBATCH --gres=gpu:1          #--gres=gpu:2 for two GPU, etc
#SBATCH --output=job_results/particle_1U6G_fbp_noise_20db.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=job_results/particle_1U6G_fbp_noise_20db.e%j    #Path and name to the file for the STDERR
#SBATCH --cpus-per-task=8     #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve

module purge
module load Miniconda3
source activate masters_thesis_marko_janic
conda list

python main.py --config run_configs/shrec_dataset_training.json
