import torch

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = torch.nn.Linear(state_dim + action_dim, 128)
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, 1)

        # Q2 architecture
        self.l4 = torch.nn.Linear(state_dim + action_dim, 128)
        self.l5 = torch.nn.Linear(128, 128)
        self.l6 = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

