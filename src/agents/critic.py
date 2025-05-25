import torch


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = torch.nn.functional.relu(self.l1(sa))
        q = torch.nn.functional.relu(self.l2(q))
        q = self.l3(q)
        return q
