import time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from config import Config, parse_args
from dataset import UKB_Dataset
from CNN import *
from CNN_Trainer import *
from learning_rate import lr_scheduler as lr

import torch
import torch.optim as optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary

args = parse_args()
config = Config(args)

ngpus = torch.cuda.device_count()
print("Number of gpus: ", ngpus)

wandb.init(project='Brain_Aging')

# Data paths and hyperparameters
dataset_pd = pd.read_csv(config.label)
print("Age distribution: ", dataset_pd['age'].describe())

# hypterparameters
BATCH_SIZE = config.batch_size
EPOCHS = config.nb_epochs
RESULTS_FOLDER = config.output
INPUT_SIZE = config.input_size
LEARNING_RATE = config.lr
OUTPUT = config.output
N_WORKERS = config.num_cpu_workers

# obtain the indices for our dataset
indices = list(range(len(dataset_pd)))
print(len(indices))

# Split indices into train and validation sets with a 3:1 ratio
train_indices, valid_indices = train_test_split(indices, test_size=0.25, random_state=7)

# Create a new dataset for training and validation
train_dataset = UKB_Dataset(config, train_indices)
valid_dataset = UKB_Dataset(config, valid_indices)

dataloader_train = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=RandomSampler(train_dataset),
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True,
                              num_workers=N_WORKERS)
dataloader_valid = DataLoader(valid_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=SequentialSampler(valid_dataset),
                              collate_fn=valid_dataset.collate_fn,
                              pin_memory=True,
                              num_workers=N_WORKERS)

# Define model and optimizer
model = CNN(in_channels=1).cuda()
# Apply the weight_initialiation
model.apply(initialize_weights)
model = torch.nn.DataParallel(model) # use with multi-gpu environment
summary(model, input_size=INPUT_SIZE, device="cuda")

# Define Optimizer
optimizer = optimizer.Adam(model.parameters(), lr=LEARNING_RATE)

# Define learning rate scheduler
if config.lr_scheduler_choice == 1:
    # 스케줄러 정의
        # 먼저 warm up: optimizer에 입력되는 learning rate = 0 또는 0에 가까운 아주 작은 값을 입력
        # T_0: 최초 주기값 (epoch/step 단위)
        # T_mult: 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값
        # eta_min: learning rate의 최소값
        # eta_max: learning rate의 최댓값
        # T_up: Warm up 시 필요한 epoch/step 수를 지정, 일반적으로 짧은 epoch/step 수를 지정
        # gamma: 주기가 반복될수록 eta_max 곱해지는 스케일값 
    scheduler = lr.CustomCosineAnnealingWarmUpRestarts(optimizer,T_0=100, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)
elif config.lr_scheduler_choice == 2:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
elif config.lr_scheduler_choice == 3:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.patience,
                                                            verbose=1, eps=1e-04, cooldown=1, min_lr=1e-6)
else:
    raise ValueError("Invalid lr_scheduler_choice. Choose between 1, 2, or 3.")

# Loss function
mse_criterion = torch.nn.MSELoss()
mae_criterion = torch.nn.L1Loss()

# Train the model
trainer = CNN_Trainer(model, 
                      RESULTS_FOLDER, 
                      dataloader_train, 
                      dataloader_valid, 
                      EPOCHS, 
                      optimizer, 
                      scheduler)

train_start = time.time()
train_mse_list, train_mae_list, valid_mse_list, valid_mae_list = trainer.train() # training!
print("train_mse_list: ", train_mse_list)
print("train_mae_list: ", train_mae_list)
print("valid_mse_list: ", valid_mse_list)
print("valid_mae_list: ", valid_mae_list)
train_end = time.time()

print(f"\nElapsed time: {(train_end - train_start) / 60:.0f} minutes")


