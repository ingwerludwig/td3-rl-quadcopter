from collections import deque
import torch
import random


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)