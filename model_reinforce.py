
import torch
from torch import nn

class Reinforce(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.layer5 = nn.Softmax()

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        return x5