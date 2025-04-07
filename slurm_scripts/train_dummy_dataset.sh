#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=train_dummy_dataset     #Name of your job
#SBATCH --time=7-00:00:00    #Maximum allocated time
#SBATCH --qos=gpu1week       #Selected queue to allocate your job (the gpu before 6hours is important, see comment: https://wiki.biozentrum.unibas.ch/display/scicore/9.+Requesting+GPUs)
#SBATCH --partition=a100-80g
#SBATCH --gres=gpu:1          #--gres=gpu:2 for two GPU, etc
#SBATCH --output=job_results/train_dummy_dataset.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=job_results/train_dummy_dataset.e%j    #Path and name to the file for the STDERR
#SBATCH --cpus-per-task=1     #Number of cores to reserve
#SBATCH --mem-per-cpu=32G     #Amount of RAM/core to reserve

module purge
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

python main.py --config run_configs/default_dummy_dataset_training.json