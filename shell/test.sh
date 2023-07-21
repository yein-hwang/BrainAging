#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=a04
#SBATCH --mem=24000MB
#SBATCH -o ./shell/cust_32_test_0.txt

source /home/username/.bashrc

echo "CustomCosineAnnealingWarmUpRestarts 적용, batch size: 32, n_workers 4"

python main_lr.py --batch_size 32 --n_workers 0 --epochs 40 --output 'model/cust_test' --ensemble_number 32 --lr 1e-6

