import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, num_layer, input_dim, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.num_layer = num_layer
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layer-2):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for i in range(self.num_layer-1):
            x = F.relu(self.fc[i](x))
        return self.fc[-1](x)