import torch.nn as nn
import torch

class DQNet(nn.Module):
    
    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1, 2), stride=1, padding=(0, 1)))
        self.dense1 = nn.Sequential(nn.Linear(20*input_dim[0]*(input_dim[1]+1), 100), 
                                nn.ReLU(),
                                nn.Linear(100, 50),
                                nn.ReLU(),
                                nn.Linear(50, output_dim))
        
        

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = torch.flatten(x1, 1)
        y = self.dense1(x1)

        return y