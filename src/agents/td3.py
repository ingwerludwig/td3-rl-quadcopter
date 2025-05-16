import torch
import numpy as np
from src.agents.actor import Actor
from src.agents.critic import Critic
from src.buffers.replay_buffer import ReplayBuffer

class TD3:
    def __init__(self, state_dim, action_dim, max_action=1):
        self.actor = Actor(state_dim, action_dim, max_action)
        self. actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00003)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00003)

        self.replay_buffer = ReplayBuffer()

        self.total_it = 0

        # Hyperparameters
        self.tau = 0.005
        self.gamma = 0.99
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.batch_size = 32

        # Mechanism to store rewards
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []

        # Mechanism to store training vals (episode wise)
        self.actor_losses_ep = []
        self.critic_losses_ep = []
        self.q_values_ep = []

    def select_action(self, state, noise=0):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
        action = self.actor(state).data.numpy().flatten()
        return action

    def train(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)

            # Compute target Q values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        # Append Q1
        self.q_values.extend(current_Q1.tolist())
        self.q_values_ep.extend(current_Q1.tolist())

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + torch.nn.functional.mse_loss(current_Q2, target_Q)

        # Append critic loss
        self.critic_losses.append(critic_loss.item())
        self.critic_losses_ep.append(critic_loss.item())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Append actor loss
            self.actor_losses.append(actor_loss.item())
            self.actor_losses_ep.append(actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    def reset(self):
        self.total_it = 0
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.replay_buffer = ReplayBuffer()

    def get_metrics_per_episode(self):
        result = np.mean(self.critic_losses_ep), np.mean(self.q_values_ep), np.mean(self.actor_losses_ep)
        self.critic_losses_ep = []
        self.actor_losses_ep = []
        self.q_values_ep = []
        return result
    
    def get_average_critic_loss(self):
        if not self.critic_losses:
            return 0
        return np.mean(self.critic_losses)
    

    def get_average_actor_loss(self):
        if not self.actor_losses:
            return 0
        return np.mean(self.actor_losses)
    

    def get_average_q_value(self):
        if not self.q_values:
            return 0
        return np.mean(self.q_values)


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
