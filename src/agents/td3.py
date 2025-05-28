import torch
import numpy as np
from src.agents.actor import Actor
from src.agents.critic import Critic
from src.buffers.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TD3 Algorithm
class TD3:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=0.01)

        self.device = device
        self.replay_buffer = ReplayBuffer()

        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.batch_size = 256
        self.total_it = 0

        # Storage for metrics
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.rewards_ep = []

        self.val_critic_losses = []
        self.val_actor_losses = []
        self.val_q_values= []

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, 0.01, size=action.shape)
            action = action + noise
        return action

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def store_reward(self, reward):
        self.rewards_ep.append(reward)

    def reset_metrics(self):
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.val_critic_losses = []
        self.val_actor_losses = []
        self.val_q_values = []

    def update(self):
        self.total_it += 1

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        self.store_reward(reward)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            # Generate target action with noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)

            # Compute target Q-values
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        # Store Q-values before update
        self.q_values.append((current_Q1.mean().item(), current_Q2.mean().item()))

        # Compute critic loss
        critic_loss1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
        critic_loss2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss1 + critic_loss2

        # Store critic losses
        self.critic_losses.append((critic_loss1.item(), critic_loss2.item()))

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # Store actor loss
            self.actor_losses.append(actor_loss.item())

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate(self, val_state, val_next_state, val_action, val_reward, val_done):
        val_state = torch.FloatTensor([val_state]).to(self.device)
        val_next_state = torch.FloatTensor([val_next_state]).to(self.device)
        val_action = torch.FloatTensor([val_action]).to(self.device)
        val_reward = torch.FloatTensor([val_reward]).to(self.device)

        val_noise = (torch.randn_like(val_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        val_next_action = (self.actor_target(val_next_state) + val_noise)

        val_target_Q1 = self.critic1_target(val_next_state, val_next_action)
        val_target_Q2 = self.critic2_target(val_next_state, val_next_action)
        val_target_Q = torch.min(val_target_Q1, val_target_Q2)
        val_target_Q = val_reward + (1 - val_done) * self.discount * val_target_Q

        val_Q1 = self.critic1(val_state, val_action)
        val_Q2 = self.critic2(val_state, val_action)

        self.val_q_values.append((val_Q1.mean().item(), val_Q2.mean().item()))

        val_critic_loss1 = torch.nn.functional.mse_loss(val_Q1, val_target_Q)
        val_critic_loss2 = torch.nn.functional.mse_loss(val_Q2, val_target_Q)
        val_critic_loss = val_critic_loss1 + val_critic_loss2

        self.val_critic_losses.append((val_critic_loss1.item(), val_critic_loss2.item()))

        val_actor_loss = -self.critic1(val_state, self.actor(val_state)).mean()
        self.val_actor_losses.append(val_actor_loss.item())

        return val_critic_loss, val_actor_loss

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

    def get_val_average_critic_loss(self):
        if not self.val_critic_losses:
            return 0
        return np.mean(self.val_critic_losses)

    def get_val_average_actor_loss(self):
        if not self.val_actor_losses:
            return 0
        return np.mean(self.val_actor_losses)

    def get_val_average_q_value(self):
        if not self.val_q_values:
            return 0
        return np.mean(self.val_q_values)


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
