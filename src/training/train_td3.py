import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.models.dynamics.quadcopter import Quadcopter
from src.controllers.lqr_quadcopter_controller import LQRQuadcopterController
from src.agents.td3 import TD3
from .train_utils import compute_next_states, save_checkpoint


def start(dataset):
    EPOCHS = 20
    ep_i = 0
    agent = TD3(state_dim=12, action_dim=16)
    quad = Quadcopter(m=1.0, Ixx=0.01, Iyy=0.01, Izz=0.02, g=9.81)
    controller = LQRQuadcopterController(quad)

    for epoch in range(EPOCHS):
        rewards = []
        for episodes in dataset:
            episode_rewards = []
            for episode in episodes[:2]:
                print(f"TrajcectType: {episode['type']}\t|\tNum sample: {episode['samples']}")
                init_state = np.zeros(12)

                for s_ref in episode['states']:
                    curr_state = init_state
                    curr_ds = curr_state - s_ref
                    qr_pair = agent.select_action(curr_ds)

                    ds, reward, ds_next, s_next = compute_next_states(qr_pair, curr_state, s_ref, quad, controller)

                    rewards.append(reward)
                    episode_rewards.append(reward)

                    agent.replay_buffer.push(ds, qr_pair, reward, ds_next, 0)

                    if len(agent.replay_buffer) >= 32:
                        agent.train()

                    curr_state = s_next

            avg_ep_reward = np.mean(episode_rewards)
            avg_ep_critic_loss, avg_ep_q, avg_ep_actor_loss = agent.get_metrics_per_episode()

            print(f"Episode: {ep_i+1}\t, Avg Episode Reward: {avg_ep_reward:.2f}\t, Avg Episode Critic Loss: {avg_ep_critic_loss:.2f}\t, Avg Episode Q: {avg_ep_q:.2f}\t, Avg Episode Actor Loss: {avg_ep_actor_loss:.2f}")

            ep_i += 1

        avg_reward = np.mean(rewards)
        avg_critic_loss = agent.get_average_critic_loss()
        avg_q = agent.get_average_q_value()
        avg_actor_loss = agent.get_average_actor_loss()

        agent.reset()

        print(f"Epoch: {epoch+1}\t, Avg Reward: {avg_reward:.2f}\t, Avg Critic Loss: {avg_critic_loss:.2f}\t, Avg Q: {avg_q:.2f}\t, Avg Actor Loss: {avg_actor_loss:.2f}")
        save_checkpoint(agent.actor, agent.critic, agent.actor_optimizer, agent.critic_optimizer, epoch)
