import torch.nn as nn
import torch

class DQNet(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.dense1 = nn.Sequential(nn.Linear(260, 500), 
                                nn.ReLU(),
                                nn.Linear(500, 200),
                                nn.ReLU(),
                                nn.Linear(200, 50),
                                nn.ReLU(),
                                nn.Linear(50, 1))
        
        

    def forward(self, x):

        y = self.dense1(x)

        return y