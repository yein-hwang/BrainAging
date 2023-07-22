import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn

import time


# Define trainer
class CNN_Trainer():
    def __init__(
            self, 
            model, 
            results_folder, 
            dataloader_train, 
            dataloader_valid, 
            epochs, 
            optimizer,
            scheduler
            ):
        super(CNN_Trainer, self).__init__()

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.epoch = 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        
        wandb.watch(self.model, log="all")

    def train(self):
        print("[ Start ]")
        start = time.time()  # Start time
        
        # Load model if exists
        latest_model_path = sorted(self.results_folder.glob('3d_cnn-*.pth.tar'), key=lambda x: int(x.stem.split('-')[1].split('.')[0]))
        if latest_model_path:
            self.load(int(latest_model_path[-1].stem.split('-')[1].split('.')[0]))
            print(f"Loaded model: {latest_model_path[-1]}")


        
        self.model.train()
        valid_loss_min = 10000
        train_mse_list, train_mae_list = [], []
        valid_mse_list, valid_mae_list = [], []
        
        while self.epoch < self.epochs:
            print(f"\nEpoch {self.epoch+1:3d}: training")
            train_mse_sum, train_mae_sum = 0, 0
            for batch_ID, (input, target) in enumerate(tqdm(self.dataloader_train)):
                input = input.cuda(non_blocking=True)
                target = target.reshape(-1, 1)
                target = target.cuda(non_blocking=True)
                
                output = self.model(input)
                
                # ----------- update -----------
                self.optimizer.zero_grad()

                mse_loss = self.mse_loss_fn(output, target)
                mae_loss = self.mae_loss_fn(output, target)
                
                mse_loss.backward() # loss_fn should be the one used for backpropagation
                
                # ----------- learning rate -----------
                self.optimizer.step()
                # learning rate update
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                    wandb.log({
                        "Learning rate update": self.scheduler.get_lr()[0]
                    })

                train_mse_sum += mse_loss.item()*input.size(0)
                train_mae_sum += mae_loss.item()*input.size(0)

            train_mse_avg = train_mse_sum / len(self.dataloader_train.dataset)
            train_mae_avg = train_mae_sum / len(self.dataloader_train.dataset)

            train_mse_list.append(train_mse_avg)
            train_mae_list.append(train_mae_avg)
            
            wandb.log({
                "Epoch": self.epoch+1,
                "Train MSE Loss": train_mse_avg,
                "Train MAE Loss": train_mae_avg
            })
            
            end = time.time()  # End time
            # Compute the duration and GPU usage
            duration = (end - start) / 60
            print(f"Epoch: {self.epoch+1}, duration for training: {duration:.2f} minutes")

            # validation step
            print(f"\nEpoch {self.epoch+1:3d}: validation")
            start = time.time()  # Start time
            self.model.eval()
            with torch.no_grad():
                valid_mse_sum, valid_mae_sum = 0, 0
                
                
                for _, (input, target) in enumerate(tqdm(self.dataloader_valid)):
                    input = input.cuda(non_blocking=True)
                    target = target.reshape(-1, 1)
                    target = target.cuda(non_blocking=True)

                    output = self.model(input)

                    mse_loss = self.mse_loss_fn(output, target) # mse_loss.item(): MSE 손실 값을 하나의 float로 변환, 각 배치에서의 평균 손실을 의미
                    mae_loss = self.mae_loss_fn(output, target)

                    valid_mse_sum += mse_loss.item()*input.size(0) # mse_loss.item() * input.size(0): 각 배치에서의 총 손실을 계산
                                                                   # input.size(0): 배치 내의 샘플 수를 반환, 이것을 MSE 손실 값에 곱하면 해당 배치의 전체 손실
                                                                   # validation set의 모든 배치를 통해 계산된 총 손실을 더하여 합산
                    valid_mae_sum += mae_loss.item()*input.size(0)

                valid_mse_avg = valid_mse_sum / len(self.dataloader_valid.dataset)
                valid_mae_avg = valid_mae_sum / len(self.dataloader_valid.dataset)

                valid_mse_list.append(valid_mse_avg)
                valid_mae_list.append(valid_mae_avg)

                    
                # learning rate update
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_mse_avg)  # Step the scheduler based on the validation loss

                wandb.log({
                    "Learning rate": self.optimizer.param_groups[0]['lr']
                })
                    
                print(f"    Epoch {self.epoch+1:2d}: training mse loss = {train_mse_avg:.3f} / validation mse loss = {valid_mse_avg:.3f}")
                print(f"    Epoch {self.epoch+1:2d}: training mae loss = {train_mae_avg:.3f} / validation mae loss = {valid_mae_avg:.3f}")
                

                # save model if better validation loss
                if valid_mse_avg < valid_loss_min:
                    print(">>>>>>>>>>>>>>>>>> Loss updates")
                    valid_loss_min = valid_mse_avg
                    self.save(self.epoch+1)
                    print(f"    Best Saved model: best-{self.results_folder}-{self.epoch+1}.pth.tar")


                wandb.log({
                    "Epoch": self.epoch+1,
                    "Validation MSE Loss": valid_mse_avg,
                    "Validation MAE Loss": valid_mae_avg,
                    "Learning rate": self.optimizer.param_groups[0]['lr']
                })


                
            self.epoch += 1
            
            end = time.time()  # End time
            # Compute the duration and GPU usage
            duration = (end - start) / 60
            print(f"Epoch: {self.epoch+1}, duration for validation: {duration:.2f} minutes")
            
        print("[ End of Epoch ]")

        self.train_mse_list = train_mse_list
        self.train_mae_list = train_mae_list
        self.valid_mse_list = valid_mse_list
        self.valid_mae_list = valid_mae_list
        
        return self.train_mse_list, self.train_mae_list, self.valid_mse_list, self.valid_mae_list
    
    def save(self, milestone):
        torch.save({"epoch": milestone-1, 
                    "state_dict": self.model.state_dict(), 
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict(),
                    "learning_rate" : self.optimizer.param_groups[0]['lr'],
                    "train_mse_list": self.train_mse_list,
                    "train_mae_list": self.train_mae_list,
                    "valid_mse_list": self.valid_mse_list,
                    "valid_mae_list": self.valid_mae_list},  
                    f"{self.results_folder}/3d_cnn-{milestone}.pth.tar")
        
    def load(self, milestone):
        checkpoint = torch.load(f"{self.results_folder}/3d_cnn-{milestone}.pth.tar")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Check the current learning rate and reset if necessary
        if self.scheduler.get_lr()[0] < 1e-6:
            # Change this to your initial learning rate
            initial_learning_rate = 0.001  
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = initial_learning_rate
            
            # Reset the scheduler with the new learning rate
            # Make sure you replace this with your scheduler's initialization code
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self.epoch = checkpoint["epoch"] + 1  # Start the next epoch after the checkpoint
        self.train_mse_list = checkpoint.get("train_mse_list", [])
        self.train_mae_list = checkpoint.get("train_mae_list", [])
        self.valid_mse_list = checkpoint.get("valid_mse_list", [])
        self.valid_mae_list = checkpoint.get("valid_mae_list", [])





