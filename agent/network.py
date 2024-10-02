import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        self.network = nn.Sequential(
           nn.Linear(in_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, obs):
        act = self.network(obs)
        return act