import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import math


class ResNets(torch.nn.Module):
    def __init__(self, backbone, head_type, num_classes=10, **kwargs):
        super(ResNets, self).__init__()
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            ValueError(f'{backbone} is not supported')

        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
        if head_type == "cls_norm":
            self.head = NormLinear(resnet.fc.in_features, num_classes)
        elif head_type == "cls":
            self.head = nn.Linear(resnet.fc.in_features, num_classes)
        else:
            raise ValueError(f'{head_type} not supported.')

    def forward(self, x):
        h = torch.squeeze(self.backbone(x))
        return h, self.head(h)


class NormLinear(nn.Module):
    def __init__(self, input, output):
        super(NormLinear, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.reset_parameters()

    def forward(self, input):
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        input_normalized = F.normalize(input, p=2, dim=1)
        output = input_normalized.matmul(weight_normalized.t())
        return output

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))