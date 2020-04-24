from torch import nn
from jcopdl.layers import conv_block, linear_block

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(3, 8),
            conv_block(8, 16),
            conv_block(16, 32),
            conv_block(32, 64),            
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            linear_block(64*4*4, 256, dropout=0.1),
            linear_block(256, 2, activation="lsoftmax")
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))