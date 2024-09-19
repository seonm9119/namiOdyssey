import torch
import torch.nn as nn

def set_train_config(cfg, model, loader):

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=cfg.epochs - cfg.lr_warmup_epochs, 
                                                           eta_min=cfg.lr_min)
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