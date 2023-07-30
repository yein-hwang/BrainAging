#!/bin/bash

#SBATCH --job-name=esb2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-10:30:00
#SBATCH --nodelist=a09
#SBATCH --mem=24000MB
#SBATCH -o ./shell/esb2_13.txt

echo "esb_2 with four gpus"
python main_cv.py --batch_size 32 --n_workers 8 --epochs 30 --output 'model/esb_' --ensemble_number 2 --lr 1e-6