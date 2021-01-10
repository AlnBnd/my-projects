from torch import nn
import match as sqrt

class L2Norm(nn.Module):
    def __init__(self, rescale):
        super(L2Norm, rescale).__init__()
        
        self.rescale = rescale
        
    def forward(self, conv):
        norm = conv.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv = conv / norm
        return self.rescale * conv
