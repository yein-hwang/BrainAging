import time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import KFold

from config import Config, parse_args
from dataset import UKB_Dataset
from CNN import *
from CNN_Trainer import *
from learning_rate import lr_scheduler as lr

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchsummary import summary


args = parse_args()
config = Config(args)

ngpus = torch.cuda.device_count()
print("Number of gpus: ", ngpus)

wandb.init(project='Brain_Aging_cv')

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


# create our k-folds
kf = KFold(n_splits=4, random_state=7, shuffle=True)
n_iter = 0
# obtain the indices for our dataset
indices = list(range(len(dataset_pd)))
print(f"Total dataset indices: {len(indices)}")

# loop over each fold
# while n_iter <= config.nb_epochs: 
for train_indices, valid_indices in kf.split(indices):

    print('\n<<< StratifiedKFold: {0}/{1} >>>'.format(n_iter+1, 4))
    
    # further split our training data into training and validation sets
    print("train_indices: ", train_indices, len(train_indices))
    print("valid_indices: ", valid_indices, len(valid_indices))
    
    # create a new dataset for this fold
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

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t_0 = int(len(dataset_pd) * 0.75 // BATCH_SIZE // 6)
    scheduler = lr.CustomCosineAnnealingWarmUpRestarts(optimizer,T_0= t_0, T_up=10, T_mult=2, eta_max=1e-3, gamma=0.5)

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
                        scheduler,
                        n_iter)


    train_start = time.time()
    trainer.train() # This will create the lists as instance variables

    # Now you can access the lists as:
    train_mse_list = trainer.train_mse_list
    train_mae_list = trainer.train_mae_list
    valid_mse_list = trainer.valid_mse_list
    valid_mae_list = trainer.valid_mae_list

    print("train_mse_list: ", train_mse_list)
    print("train_mae_list: ", train_mae_list)
    print("valid_mse_list: ", valid_mse_list)
    print("valid_mae_list: ", valid_mae_list)
    train_end = time.time()

    print(f"\nElapsed time for one epoch in cv: {(train_end - train_start) / 60:.0f} minutes")
    # print('<<< Stratified {0} Fold mean MSE: {1:.4f}, std: {2:.4f}'.format(n_iter, np.mean(valid_mse_list), np.std(valid_mse_list)))
    # print('<<< Stratified {0} Fold mean MAE: {1:.4f}, std: {2:.4f}'.format(n_iter, np.mean(valid_mae_list), np.std(valid_mae_list)))
    
    n_iter += 1





