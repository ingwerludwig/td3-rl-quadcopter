import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
from src.agents.td3 import TD3
from .train_utils import save_checkpoint, save_metrics_to_json, delete_existing, create_metrics_episode
from src.visualization.visualization_utils import plot_training_metrics
from src.envs.quadcopter_lqr_env import QuadcopterLQREnv
from src.models.dynamics.quadcopter import Quadcopter
from src.config.constant import MAX_STEP_PER_EPISODE
from src.training.train_utils import log_print
from src.visualization.generate_visualization import generate_quad_sim

def start(dataset, val_dataset):
    quadcopter = Quadcopter()
    env = QuadcopterLQREnv(quadcopter=quadcopter)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_step_per_episode = MAX_STEP_PER_EPISODE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_dim, action_dim, device)

    # delete_existing(subdir_names=["results", "logs"])

    avg_ep_rewards = []
    print("START TRAINING ...")

    # generate_quad_sim(lqr_quad_env=env,
    #                   quadcopter=quadcopter,
    #                   initial_state=state,
    #                   reference_state=env.ref_states,
    #                   t_total=10,
    #                   episode=ep_i,
    #                   sample=sample_i,
    #                   save_filename="plot.png")


    for ep_i, episode in enumerate(zip(dataset, val_dataset)):
        agent.load_checkpoint(
            f"/home/citiai-cygnus/VisionRAG-Ingwer/TugasIngwer/TugasAIForRobotics/src/FIXED RESULT/CIRCLE/checkpoint/td3_quadcopter_epoch_{ep_i+1}.pth")
        train_episode, val_episode = episode
        episode_reward = 0
        rewards = []
        res_step = []
        state, _ = env.reset(np.zeros(12))
        # for sample_i, ref_state in enumerate(train_episode['states']):
        #     done = False
        #     sample_reward = 0
        #     env.set_ref_states(ref_state)
        #
        #     log_print(f"Initial state:\t\t{np.round(state, 2)}")
        #     log_print(f"Ref state:\t\t{np.round(env.ref_states, 2)}\n")
        #
        #     # generate_quad_sim(lqr_quad_env = env,
        #     #                   quadcopter=quadcopter,
        #     #                   initial_state=state,
        #     #                   reference_state=env.ref_states,
        #     #                   t_total=10,
        #     #                   episode=ep_i,
        #     #                   sample=sample_i,
        #     #                   save_filename="plot.png")
        #
        #     for i in range(max_step_per_episode):
        #         lqr_params = agent.select_action(state - ref_state)
        #         env.update_lqr_params(lqr_params)
        #
        #         action = env.compute_quad_actions_from_controller()
        #         next_state, reward, done, _, info = env.step()
        #         agent.add_to_buffer(state - ref_state, lqr_params, reward, next_state - ref_state, done)
        #
        #         episode_reward += reward
        #         sample_reward += reward
        #         state = next_state
        #
        #         if len(agent.replay_buffer.storage) >= agent.batch_size:
        #             agent.update()
        #
        #         if done:
        #             log_print(info['status'])
        #             break
        #
        #     rewards.append(episode_reward)
        #     log_print(f"Sample: {sample_i + 1}, Reward: {sample_reward:.2f}")
        #
        # metrics_episode = create_metrics_episode(ep_i, np.mean(rewards), agent.get_average_critic_loss(),agent.get_average_q_value(),agent.get_average_actor_loss())
        # save_metrics_to_json(metrics_episode,f"Episode_{ep_i+1}_training_metrics.json")
        # plot_training_metrics(ep_i+1, save_dir=f"epoch_plot.png")
        # save_checkpoint(agent, ep_i)
        #
        # agent.reset_metrics()

        print("START EVALUATING ...")
        val_episode_reward = 0
        val_rewards = []
        val_state, _ = env.reset(np.zeros(12))

        for val_sample_i, val_ref_state in enumerate(val_episode['states']):
            env.set_ref_states(val_ref_state)
            lqr_params = agent.select_action(env.states - env.ref_states)
            env.update_lqr_params(lqr_params)
            _ = env.compute_quad_actions_from_controller(val_state, val_ref_state)
            val_next_state, val_reward, val_done, _, info = env.step()
            print(np.round(val_next_state, 2))
            print(np.round(val_ref_state, 2))
            print(info)
            val_episode_reward += val_reward
            agent.evaluate(val_state-val_ref_state, val_next_state-val_ref_state, lqr_params, val_reward, val_done)
            val_state = val_next_state
            val_rewards.append(val_episode_reward)
        metrics_episode = create_metrics_episode(ep_i, np.mean(val_rewards), agent.get_val_average_critic_loss(),
                                                 agent.get_val_average_q_value(), agent.get_val_average_actor_loss())
        save_metrics_to_json(metrics_episode, f"Episode_{ep_i + 1}_validation_metrics.json")
        plot_training_metrics(ep_i + 1, save_dir=f"validation_epoch_plot.png")
        agent.reset_metrics()
