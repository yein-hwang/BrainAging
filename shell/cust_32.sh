#!/bin/bash

#SBATCH --job-name=cust_32
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=b04
#SBATCH --mem=24000MB
#SBATCH -o ./shell/cust_32_1.txt

source /home/username/.bashrc

echo "CustomCosineAnnealingWarmUpRestarts 적용, batch size: 32, n_workers: 8"

python main_lr.py --batch_size 32 --n_workers 8 --epochs 40 --output 'model/cust_' --ensemble_number 32 --lr 1e-6

