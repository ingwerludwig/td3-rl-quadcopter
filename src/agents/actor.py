import torch
import numpy as np


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.nn.functional.relu(self.l1(state))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.softplus(self.l3(x))
        return torch.clamp(x, 1, np.inf)
