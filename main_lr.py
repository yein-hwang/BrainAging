import time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from config import Config, parse_args
from dataset import UKB_Dataset
from CNN import *
from CNN_Trainer_lr import *
from lr_scheduler import CustomCosineAnnealingWarmUpRestarts

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary

args = parse_args()
config = Config(args)

ngpus = torch.cuda.device_count()
print("Number of gpus: ", ngpus)

wandb.init(project='lr_test')

# Data paths and hyperparameters
dataset_pd = pd.read_csv(config.label)
print("Age distribution: ", dataset_pd['age'].describe())

# hypterparameters
BATCH_SIZE = config.batch_size
EPOCHS = config.nb_epochs
RESULTS_FOLDER = config.output
INPUT_SIZE = config.input_size
LEARNING_RATE = config.lr
WARMUP_EPOCHS = 10
OUTPUT = config.output
N_WOKERS = config.num_cpu_workers




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
                              num_workers=N_WOKERS)
dataloader_valid = DataLoader(valid_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=SequentialSampler(valid_dataset),
                              collate_fn=valid_dataset.collate_fn,
                              pin_memory=True,
                              num_workers=N_WOKERS)


# Define model and optimizer
model = CNN(in_channels=1).cuda()
# Apply the weight_initialiation
model.apply(initialize_weights)
model = torch.nn.DataParallel(model) # use with multi-gpu environment
summary(model, input_size=INPUT_SIZE, device="cuda")

# 최적화기 정의
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 스케줄러 정의
    # 먼저 warm up: optimizer에 입력되는 learning rate = 0 또는 0에 가까운 아주 작은 값을 입력
    # T_0: 최초 주기값
    # T_mult: 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값
    # eta_min: learning rate의 최소값
    # eta_max: learning rate의 최댓값
    # T_up: Warm up 시 필요한 epoch 수를 지정, 일반적으로 짧은 epoch 수를 지정
    # gamma: 주기가 반복될수록 eta_max 곱해지는 스케일값 
    # 아래 setting -> 5 epoch 주기를 초깃값으로 가지되 반복될수록 주기를 2배씩 늘리는 방법
scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=3, T_mult=2, eta_max=1e-4, T_up=5, gamma=0.5)

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


