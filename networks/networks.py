import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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

