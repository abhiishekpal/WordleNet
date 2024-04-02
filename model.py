import torch.nn as nn
import torch

class DQNet(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.dense1 = nn.Sequential(nn.Linear(260, 1500), 
                                nn.ReLU(),
                                nn.Linear(1500, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1))
        
        

    def forward(self, x):

        y = self.dense1(x)

        return y