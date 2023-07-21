#!/bin/bash

#SBATCH --job-name=cos
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=a09
#SBATCH --mem=24000MB
#SBATCH -o ./shell/cos_1.txt


echo "CosineLR 적용, batch size: 32, n_workers: 8"
python main.py --batch_size 32 --n_workers 8 --epochs 40 --output 'model/cos_' --ensemble_number 32 --lr 1e-6 --lr_scheduler_choice 2 