#!/bin/bash

#SBATCH --job-name=esb1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --nodelist=a07
#SBATCH --time=0-12:00:00
#SBATCH --mem=24000MB
#SBATCH -o ./shell/esb1/esb1_cv2_1.txt

echo "esb_1 with four gpus"
python main_cv.py --batch_size 32 --n_workers 8 --epochs 40 --output 'model/esb_' --ensemble_number 1 --lr 1e-6