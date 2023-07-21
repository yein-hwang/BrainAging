import time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from config import Config, parse_args
from dataset import UKB_Dataset
from CNN import *
from CNN_Trainer import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

args = parse_args()
config = Config(args)

ngpus = torch.cuda.device_count()
print("Number of gpus: ", ngpus)

wandb.init(project='test')

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




start_time = time.time()

# Define model and optimizer
model = CNN(in_channels=1).cuda()
# Apply the weight_initialiation
model.apply(initialize_weights)
model = torch.nn.DataParallel(model) # use with multi-gpu environment
summary(model, input_size=INPUT_SIZE, device="cuda")

end_time = time.time()
print("Time taken to define and initialize model: ", end_time - start_time)

start_time = time.time()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.patience,
#                                                  verbose=1, eps=1e-04, cooldown=1, min_lr=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-8)


end_time = time.time()
print("Time taken to define optimizer and scheduler: ", end_time - start_time)

# Loss function
mse_criterion = torch.nn.MSELoss()
mae_criterion = torch.nn.L1Loss()


# obtain the indices for our dataset
indices = list(range(len(dataset_pd)))
print(len(indices))



# Split indices into train and validation sets with a 3:1 ratio
train_indices, valid_indices = train_test_split(indices, test_size=0.25, random_state=7)

# Create a new dataset for training and validation
train_dataset = UKB_Dataset(config, train_indices)
valid_dataset = UKB_Dataset(config, valid_indices)

n_workers = 4 * ngpus
# n_workers = BATCH_SIZE / ngpus

dataloader_train = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=RandomSampler(train_dataset),
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True,
                              num_workers=n_workers)
dataloader_valid = DataLoader(valid_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=RandomSampler(valid_dataset),
                              collate_fn=valid_dataset.collate_fn,
                              pin_memory=True,
                              num_workers=n_workers)

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
print("train_mae_list: ", train_mse_list)
print("valid_mse_list: ", valid_mse_list)
print("valid_mae_list: ", valid_mae_list)
train_end = time.time()

print(f"\nElapsed time: {(train_end - train_start) / 60:.0f} minutes")
