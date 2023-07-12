import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class UKB_Dataset(Dataset):
    def __init__(self, config, indices=None):
        super(UKB_Dataset, self).__init__()
        self.config = config
        self.data_dir = config.data
        self.data_csv = pd.read_csv(config.label)
        
        self.image_names = [self.data_csv['id'][i] for i in indices]
        self.labels = [self.data_csv['age'][i] for i in indices]
        
        print("images_names: ", len(self.image_names), self.image_names[-1])
        print("labels: ", len(self.labels), self.labels[-1])
    

        self.transform = T.Compose([
            # T.Resize((image_size, image_size, image_size))
            T.ToTensor()
        ])
    def collate_fn(self, batch):
        images, labels = zip(*batch)  # separate images and labels
        images = torch.stack(images)  # stack images into a tensor
        labels = torch.tensor(labels)  # convert labels into a tensor
        return images, labels

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]

        # Load the image
        image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.float32)
        image = torch.from_numpy(image).float()
        image = image.permute(3, 0, 1, 2)

        np.random.seed()
        age = torch.tensor(label, dtype=torch.float32)

        return (image, age)
    
    
    
class ADNI_Dataset(Dataset):
    def __init__(self, image_path, label_path, image_size=224):
        super(ADNI_Dataset, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        
        image_names, labels = [], []
        
        with open(label_path, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # skip the header
            for line in csv_reader:
                subject_id = line[1]  # id column

                image_name = f'sub-{subject_id}X{date_str}'
                image_names.append(image_name)
                labels.append(float(line[3]))  # age column
        
        self.image_names = image_names
        self.labels = labels
        
        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)
        
        self.input_size = (1, 80, 80, 80)
        
    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        image_file_path = os.path.join(self.image_path, image_name, 'brain_to_MNI_nonlin.nii.gz')
        
        # Load the image using nibabel
            # 'nibabel' to load '.nii.gz' files, which are specific to brain imaging data
        image = nib.load(image_file_path)
        image = np.swapaxes(image.get_fdata(),1,2)
        image = np.flip(image,1)
        image = np.flip(image,2)
        image = resize(image, (self.input_size[1], self.input_size[2], self.input_size[3]), mode='constant')
        image = image.reshape(self.input_size[0], self.input_size[1], self.input_size[2], self.input_size[3])  # Add extra dimension for single channel  # Add extra dimension for single channel

        np.random.seed()
        x = self.transforms(image)
        
        return (x, label)