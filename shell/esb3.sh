#!/bin/bash

#SBATCH --job-name=esb3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-11:00:00
#SBATCH --nodelist=a04
#SBATCH --mem=24000MB
#SBATCH -o ./shell/esb3/esb3_cv2_3.txt

echo "esb_3 with four gpus, epoch 38~40 트레이닝"
python main_cv.py --batch_size 32 --n_workers 8 --epochs 40 --output 'model/esb_' --ensemble_number 3 --lr 1e-6