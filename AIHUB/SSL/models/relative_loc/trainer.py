import torch
import torch.nn as nn
from namiOdyssey.AIHUB.SSL.models.relative_loc.utils import build_loader
from namiOdyssey.AIHUB.SSL.models.relative_loc.model import RelativeLoc
from namiOdyssey.utils.trainer import Trainer
import time


def set_train_config(cfg, model):

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_config = {'device': cfg.device,
              'epochs': cfg.epochs,
              'optimizer': optimizer,
              'criterion': nn.CrossEntropyLoss(),
              'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)}
    
    cfg.logger.info(f'Train Configs: {train_config}')
    return train_config


class RelativeLocTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__()

        
        self.model = RelativeLoc()
        self.loader = build_loader(cfg)

        self.cfg = cfg
        self.logger = cfg.logger
        config = set_train_config(cfg, self.model)

        self.global_tr_losses = []
        self.global_val_losses = []

        for key, value in config.items():
            setattr(self, key, value)

    def run(self):
        
        for epoch in range(self.epochs):
            start_time = time.time()

            self.model = self.model.to(self.device)
            tr_losses = self.train(model=self.model,
                                   loader=self.loader['train'],
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   device=self.device)
            
            val_losses, total, correct = self.cls_eval(model=self.model,
                                                        loader=self.loader['val'],
                                                        criterion=self.criterion,
                                                        device=self.device)

            self.logger.info(f"Val Progress --- total:{total}, correct:{correct.item()}")
            self.logger.info(f"Val Accuracy of the network on the {total} test images: {(100 * correct / total):.2f}%")


            self.global_tr_losses.append(sum(tr_losses) / len(tr_losses))
            self.global_val_losses.append(sum(val_losses) / len(val_losses))

            self.scheduler.step(self.global_val_losses[-1])
            self.logger.info(f"Epoch [{epoch + 1}/{self.epochs}], TRNLoss:{self.global_tr_losses[-1]:.4f}, VALoss:{self.global_val_losses[-1]:.4f}, Time:{(time.time() - start_time) / 60:.2f}")

            


