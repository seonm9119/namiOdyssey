import torch
import torch.nn as nn
from tqdm import tqdm


def set_train_config(cfg, model, loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    train_config = {'device': cfg.device,
                    'epochs': cfg.epochs,
                    'model' : model,
                    'loader' : loader,
                    'optimizer': optimizer,
                    'criterion': nn.CrossEntropyLoss(),
                    'scheduler': scheduler,
                    'save_interval': cfg.save_interval,
                    'save_path': cfg.save_path}
    

    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint)
        train_config['model'].load_state_dict(checkpoint['model'])
        train_config['optimizer'].load_state_dict(checkpoint['optimizer'])
        train_config['scheduler'].load_state_dict(checkpoint['scheduler'])
        train_config['epochs'] = checkpoint['epoch']
        print(f"load checkpoint")
    
    print(f'Train Configs: {train_config}')
    return train_config



def to_device(data, device):
    for key, value in data.items():
        data[key] = value.to(device)
    return data



def save(epoch, model, optimizer, scheduler, path):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
        
    torch.save(state, path)

def train(model, loader, optimizer, criterion, device):
    
    losses = []
    model.train()
    for data in tqdm(loader):
        data = to_device(data, device)

        optimizer.zero_grad()
        output = model(data['input'])
        loss = criterion(output, data['label'])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


def eval(model, loader, criterion, device):
        
    losses = []
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data in tqdm(loader):
            data = to_device(data, device)

            output = model(data['input'])
            loss = criterion(output, data['label'])
            losses.append(loss.item())

            predicted = torch.argmax(output.data, dim=1)
            total += data['label'].size(0)
            correct += (predicted == data['label']).sum()
        
    return losses, total, correct