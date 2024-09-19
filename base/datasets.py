import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BasetDataset(Dataset):
    def __init__(self,
                 file_dir='train.csv',
                 transform=None):
        
        df = pd.read_csv(file_dir, usecols=['image_path', 'label'])
        self.img, self.label = df['image_path'], df['label']
        self.transform = transform

    def __len__(self):
        return len(self.img)
    

    def __getitem__(self, idx):
        img = Image.open(self.img[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.label[idx], dtype=torch.long)
        return {'input': img, 'label': label}
    