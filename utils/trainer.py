import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        pass

    def train(self, model, loader, optimizer, criterion, device):

        losses = []
        model.train()
        for data in tqdm(loader):

            for key, value in data.items():
                data[key] = value.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data['label'])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        return losses
    
    def cls_eval(self, model, loader, criterion, device):
        
        losses = []
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(loader):
                
                for key, value in data.items():
                    data[key] = value.to(device)
            
                output = model(data)
                loss = criterion(output, data['label'])

                losses.append(loss.item())

                predicted = torch.argmax(output.data, dim=1)
                total += data['label'].size(0)
                correct += (predicted == data['label']).sum()
        
        return losses, total, correct
    

    def save(self, model, dir):
        torch.save(model.state_dict(), 'model_weights.pth')
