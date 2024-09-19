
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def plot_train_val_loss(trnloss, valloss):
    plt.plot(range(len(trnloss)), trnloss, label='Train Loss')
    plt.plot(range(len(valloss)), valloss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Main Network Training/Validation Loss plot')
    plt.legend()
    plt.show()



def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)  
    std = torch.tensor(std).view(1, 3, 1, 1)
    image = image * std + mean                  
    return image


def imshow(images, grid_size=(10, 10)):

    max_len = grid_size[0] * grid_size[1]
    images = images if images.size(0) < max_len else images[:max_len]
    images = denormalize(images)
    

    grid_img = make_grid(images, nrow=grid_size[1], padding=2)  
    grid_img = grid_img.permute(1, 2, 0)  

    
    plt.figure(figsize=(15, 15))
    plt.imshow(torch.clip(grid_img, 0, 1))
    plt.axis('off')
    plt.show()