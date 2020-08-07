import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
class Measurement_Matrix(nn.Module):
    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False
        
        mask = torch.ones([1, 3, 1024, 1024], dtype=torch.float32).type('torch{}.FloatTensor'.format(self.cuda))
        data = np.load('mask.bin.npy')
        for i in range(1024):
            for j in range(1024):
                if(data[i][j] == 0):
                    mask[:,:,i,j] = 0

        self.mask = mask

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        #redefine forward operation to just be a masking image operation
        return torch.mul(self.mask,x)
