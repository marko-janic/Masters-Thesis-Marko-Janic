#!/bin/bash
#The previous line is mandatory

#SBATCH --job-name=number_of_training_volumes_ablation_study    #Name of your job
#SBATCH --time=7-00:00:00    #Maximum allocated time
#SBATCH --qos=gpu1week      #Selected queue to allocate your job (the gpu before 6hours is important, see comment: https://wiki.biozentrum.unibas.ch/display/scicore/9.+Requesting+GPUs)
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1          #--gres=gpu:2 for two GPU, etc
#SBATCH --output=job_results/number_of_training_volumes_ablation_study.o%j   #Path and name to the file for the STDOUT
#SBATCH --error=job_results/number_of_training_volumes_ablation_study.e%j    #Path and name to the file for the STDERR
#SBATCH --cpus-per-task=16     #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve

module purge
module load Miniconda3
source activate masters_thesis_marko_janic
conda list

python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_1_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_2_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_3_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_4_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_5_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_6_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_7_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_8_training_volumes.json
python main.py --config run_configs/ablation_studies/number_of_training_volumes/fbp_20dB_9_training_volumes.json
