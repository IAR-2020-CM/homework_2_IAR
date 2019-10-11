import torch
import math
from torch import nn
from matplotlib import pyplot

class MLP(nn.Module):

    def __init__(self, n_neurones=128, e_dim=1, o_dim=4):
        super().__init__()

        self.linear_in_to_hidden = nn.Linear(e_dim, n_neurones)
        self.linear_hidden_to_out = nn.Linear(n_neurones, o_dim)

    def forward(self, x):
        h = self.linear_in_to_hidden(x)
        h = nn.functional.relu(h)
        return self.linear_hidden_to_out(h)

