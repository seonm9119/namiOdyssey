from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

def image_to_patches(img, jitter=7):
    
    # Define the transform to convert PIL image to tensor
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    split_per_side = 3 
    patch_jitter = split_per_side * jitter

    _, h, w = img.shape
    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    assert h_patch > 0 and w_patch > 0


    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            p = TF.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
            p = transforms.RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    
    return patches

class RelativeLocDataset(Dataset):
    def __init__(self,
                 file_dir='train.csv', 
                 patch_size=15, 
                 jitter=3,
                 transform=None):
    
        self.patch_size, self.jitter = patch_size, jitter
        self.transform = transform

        df = pd.read_csv(file_dir, usecols=['image_path'])
        self.img = df['image_path']

    def __len__(self):
        return len(self.img)
    

    def __getitem__(self, idx):
        img = Image.open(self.img[idx]).convert('RGB')
        patches = image_to_patches(img, jitter=self.jitter)
        
        # create a list of patch pairs
        uniform_patch = torch.stack([patches[4] for _ in range(8)], dim=0)
        random_patch = torch.stack([patches[i] for i in range(9) if i != 4], dim=0)
        position_label = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])

        if self.transform is not None:
            random_patch = self.transform(random_patch)
            uniform_patch = self.transform(uniform_patch)
       
        return uniform_patch, random_patch, position_label
      
    
  



  
  