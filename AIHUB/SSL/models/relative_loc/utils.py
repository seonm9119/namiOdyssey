import torch
from torchvision import transforms
from namiOdyssey.AIHUB.SSL.models.relative_loc.datasets import RelativeLocDataset



def _collate_fn(batch):

    uniform, random, label = zip(*batch)  # Separate data and labels
    uniform = torch.stack(uniform)  # Combine the list of tensors into a single tensor
    random = torch.stack(random)

    # Reshape from [B, 8, C, patch_size, patch_size] to [B*8, C, patch_size, patch_size]
    uniform = uniform.flatten(start_dim=0, end_dim=1)
    random = random.flatten(start_dim=0, end_dim=1)
    label = torch.cat(label)

    return {'uniform':uniform, 'random': random, 'label': label}


def _build_loader(cfg, file_path, transform, key):

    dataset = RelativeLocDataset(file_dir=file_path, 
                                 patch_size=cfg.patch_size, 
                                 jitter=cfg.jitter,
                                 transform=transform)
    
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=cfg.batch_size,
                                       shuffle= True if key == 'train' else False,
                                       collate_fn=_collate_fn,
                                       num_workers=cfg.num_workers)



def build_loader(cfg):
    return LOADER[cfg.data](cfg)


def tiny_imagenet(cfg):
    from namiOdyssey.datasets.data_sources.tiny_imagenet import SPLIT_FUNC, check_file

    transform = transforms.Compose([#transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])
    
    loader = {}
    for key in SPLIT_FUNC.keys():
        file_path = check_file(cfg.data_dir, key)
        loader[key] = _build_loader(cfg, file_path, transform, key)

    return loader

LOADER = {'tiny-imagenet': tiny_imagenet}