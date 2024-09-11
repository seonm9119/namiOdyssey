import torch
from torchvision import transforms
from namiOdyssey.AIHUB.SSL.models.relative_loc.datasets import RelativeLocDataset
from namiOdyssey.datasets.data_sources.tiny_imagenet import make_train_dataframe, make_val_dataframe
from namiOdyssey.datasets.utils import check_file



def _collate_fn(batch):

    uniform, random, label = zip(*batch)  # Separate data and labels
    uniform = torch.stack(uniform)  # Combine the list of tensors into a single tensor
    random = torch.stack(random)

    # Reshape from [B, 8, C, patch_size, patch_size] to [B*8, C, patch_size, patch_size]
    uniform = uniform.flatten(start_dim=0, end_dim=1)
    random = random.flatten(start_dim=0, end_dim=1)
    label = torch.cat(label)

    return {'uniform':uniform, 'random': random, 'label': label}


def _build_loader(args, file_dir, transform, train=True):

    
    dataset = RelativeLocDataset(file_dir=file_dir, 
                                 patch_size=args.patch_size, 
                                 jitter=args.jitter,
                                 transform=transform)
    
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=args.batch_size,
                                       shuffle=train,
                                       collate_fn=_collate_fn,
                                       num_workers=args.num_workers)
                                        

def build_loader(cfg):

    transform = transforms.Compose([#transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])
    
    train_file = check_file(cfg.data_dir, 'train.csv', make_train_dataframe)
    val_file = check_file(cfg.data_dir, 'val.csv', make_val_dataframe)

    return {'train': _build_loader(cfg, train_file, transform, train=True),
            'val': _build_loader(cfg, val_file, transform, train=False)}
