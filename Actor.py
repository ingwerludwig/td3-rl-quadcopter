import torch


class Actor(torch.nn.Module):

    def __init__(self, state_dim, action_dim, max_action=1) -> None:
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, action_dim)
        self.softplus = torch.nn.Softplus()
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = self.softplus(self.l3(a)) + 0.0001
        return self.max_action * a
    