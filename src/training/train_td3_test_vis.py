import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import math
from src.agents.td3 import TD3
from .train_utils import save_checkpoint, save_metrics_to_json, delete_existing, create_metrics_episode
from src.visualization.visualization_utils import plot_training_metrics
from src.envs.quadcopter_lqr_env import QuadcopterLQREnv
from src.models.dynamics.quadcopter import Quadcopter
from src.config.constant import MAX_STEP_PER_EPISODE
from src.training.train_utils import log_print
from src.visualization.generate_visualization import generate_quad_sim, generate_video_plot

def start(dataset, val_dataset):
    quadcopter = Quadcopter()
    env = QuadcopterLQREnv(quadcopter=quadcopter)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_step_per_episode = MAX_STEP_PER_EPISODE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_dim, action_dim, device)

    agent.load_checkpoint("/home/citiai-cygnus/VisionRAG-Ingwer/TugasIngwer/TugasAIForRobotics/src/FIXED RESULT/CIRCLE/checkpoint/td3_quadcopter_epoch_100.pth")
    result = []

    for ep_i, episode in enumerate(zip(dataset, val_dataset)):
        train_episode, val_episode = episode
        state, _ = env.reset(np.zeros(12))
        env.states = np.zeros(12)
        histories = []

        for sample_i, ref_state in enumerate(val_episode['states']):
            env.set_ref_states(ref_state)
            print(f"States: \t\t{np.round(env.states, 2)}")
            print(f"Ref St: \t\t{np.round(env.ref_states, 2)}")
            lqr_params = agent.select_action(env.states-env.ref_states)
            env.update_lqr_params(lqr_params)
            _ = env.compute_quad_actions_from_controller()
            val_next_state, val_reward, val_done, _, info = env.step()
            histories.append(np.copy(env.states))

        generate_quad_sim(state=histories,
                          reference_state=val_episode['states'],
                          episode=ep_i,
                          save_filename="plot.png")


        print("Generating Video...")
        generate_video_plot(
            states=histories,
            ref_states=val_episode['states'],
            save_file_name="Line1 Sim.mp4"
        )

        break
