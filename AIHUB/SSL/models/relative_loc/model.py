import torch
import torch.nn as nn
from namiOdyssey.AIHUB.SSL.models.relative_loc.resnet import resnet18


class RelativeLoc(nn.Module):
    def __init__(self, backbone='resnet18', 
                 dim=[512, 4096], 
                 n_classes=8):
        super(RelativeLoc, self).__init__()
        

        self.backbone = resnet18()

        in_dim, out_dim = dim[0], dim[1]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc7 = nn.Sequential(nn.Linear(in_dim * 2, out_dim),
                                 nn.BatchNorm1d(out_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout())
        
        self.fc_cls = nn.Linear(out_dim, n_classes)


    def forward(self, x):

        patch_1 = self.backbone(x['uniform'])
        patch_2 = self.backbone(x['random'])

        patch_cat = torch.cat((patch_1, patch_2), dim=1)
        output = self.avg_pool(patch_cat)

        output = output.view(output.size(0), -1)
        output = self.fc7(output)
        output = self.fc_cls(output)

        return output
        