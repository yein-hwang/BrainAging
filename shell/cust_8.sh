#!/bin/bash

#SBATCH --job-name=cust_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=a04
#SBATCH --mem=24000MB
#SBATCH --cpus-per-task=32
#SBATCH -o ./shell/cust_8_5.txt

source /home/username/.bashrc

echo "CustomCosineAnnealingWarmUpRestarts 적용, batch size: 8"

python main_lr.py --batch_size 8 --epochs 40 --output 'model/cust_' --ensemble_number 8 --lr 1e-6

