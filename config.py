import os
import argparse

class Config:

    def __init__(self, args):
        
        for name, value in vars(args).items():
            setattr(self, name, value)
       
        self.data = '/media/leelabsg-storage1/DATA/UKBB/bulk/20252_numpy/20252_individual_samples' 
        self.label = './csv/ukbb_cn.csv' 
        self.input_size = (1, 128, 128, 128) 
        self.batch_size = args.batch_size
        self.pin_mem = True
        self.num_cpu_workers = args.n_workers
        self.cuda = True
        
        self.ensemble_number = args.ensemble_number
        # self.model = 'DenseNet' # 'UNet
        self.nb_epochs = args.epochs
        self.lr = args.lr # Optimizer
        self.weight_decay = 5e-5
        self.patience = 1
        # self.tf = 'cutout' 
        
        self.output = os.path.join(str(args.output) + str(args.ensemble_number))
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ensemble_number', type=int, default=1)
    parser.add_argument('--output', type=str, default='./model')
    parser.add_argument('--lr', type=float, default=0.0001)

    return parser.parse_args()