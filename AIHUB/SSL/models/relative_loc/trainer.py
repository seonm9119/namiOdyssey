import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from namiOdyssey.AIHUB.SSL.models.relative_loc.utils import build_loader
from namiOdyssey.AIHUB.SSL.models.relative_loc.model import RelativeLoc
from namiOdyssey.base.trainer.trainer import Trainer
from namiOdyssey.base.trainer.utils import to_device, save


def set_train_config(cfg, model, loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)
    
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



class RelativeLocTrainer(Trainer):
    def __init__(self, config):
        super().__init__()

        train_config = set_train_config(config, 
                                        model=RelativeLoc(), 
                                        loader=build_loader(config))
        
        self.set_config(train_config)
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, model, loader, optimizer, criterion, device):
        losses = []
        model.train()
        
        for data in tqdm(loader):
            data = to_device(data, device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data['label'])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        return losses
    

    def eval(self, model, loader, criterion, device):
        
        losses = []
        correct = 0
        total = 0
        model.eval()
    
        with torch.no_grad():
            for data in tqdm(loader):
                data = to_device(data, device)

                output = model(data)
                loss = criterion(output, data['label'])
                losses.append(loss.item())

                predicted = torch.argmax(output.data, dim=1)
                total += data['label'].size(0)
                correct += (predicted == data['label']).sum()
        
        return losses, total, correct
    

    def run(self):

        for epoch in range(self.epochs):
            start_time = time.time()

            self.model = self.model.to(self.device)
            tr_losses = self.train(model=self.model,
                                   loader=self.loader['train'],
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   device=self.device)
            
            val_losses, total, correct = self.eval(model=self.model,
                                                   loader=self.loader['val'],
                                                   criterion=self.criterion,
                                                   device=self.device)

            self.global_tr_losses.append(sum(tr_losses) / len(tr_losses))
            self.global_val_losses.append(sum(val_losses) / len(val_losses))

            self.logger.info(f"Val Progress --- total:{total}, correct:{correct.item()}")
            self.logger.info(f"Val Accuracy of the network on the {total} test images: {(100 * correct / total):.2f}%")

            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}], "
                        f"TRNLoss: {self.global_tr_losses[-1]:.4f}, "
                        f"VALoss: {self.global_val_losses[-1]:.4f}, "
                        f"Time: {(time.time() - start_time) / 60:.2f}")
            
            if (epoch + 1) % self.save_interval == 0 or epoch == self.epochs - 1:
                file_path = os.path.join(self.save_path, f"epoch_{epoch+1}.pth")
                save(epoch, self.model, self.optimizer, self.scheduler, file_path)
                self.logger.info(f"Model saved at epoch {epoch+1} to {file_path}")

            if self.scheduler is not None:
                self.scheduler.step(self.global_val_losses[-1])
            


