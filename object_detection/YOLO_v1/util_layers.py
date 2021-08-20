import torch
import torch.nn as nn
import torch.nn.functional as F

# Utils

class squeeze_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()

class flatten_layer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)