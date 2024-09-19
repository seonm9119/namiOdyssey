import torch
from tqdm import tqdm

def to_device(data, device):
    for key, value in data.items():
        data[key] = value.to(device)
    return data


def save(epoch, model, optimizer, scheduler, trnloss, valloss, path):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'trnloss': trnloss,
             'valloss': valloss}
        
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