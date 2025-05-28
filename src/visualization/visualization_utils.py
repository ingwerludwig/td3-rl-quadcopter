import copy
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math

from matplotlib.figure import Figure
from scipy.integrate import odeint

from src.envs.quadcopter_lqr_env import QuadcopterLQREnv
from src.models.dynamics.quadcopter import Quadcopter
from src.training.train_utils import log_print


def plot_results(drone_history, ref_states, episode, filename, trajectory_name: str = "Unknown",show=True):
    # Extract states
    ref_states = np.array(ref_states)
    drone_history = np.array(np.round(drone_history, 2))
    x_ref, y_ref, z_ref = ref_states[:, 0], ref_states[:, 2], ref_states[:, 4]
    x_hist, y_hist, z_hist = drone_history[:, 0], drone_history[:, 2], drone_history[:, 4]
    start_x, start_y, start_z = drone_history[0][0], drone_history[0][2], drone_history[0][4]
    end_x, end_y, end_z = drone_history[-1][0], drone_history[-1][2], drone_history[-1][4]

    print(drone_history.shape, ref_states.shape)
    x_err = np.sqrt((x_ref - x_hist)**2)
    y_err = np.sqrt((y_ref - y_hist)**2)
    z_err = np.sqrt((z_ref - z_hist)**2)

    print(f"This is percentage error x {np.mean(x_err)}")
    print(f"This is percentage error y {np.mean(y_err)}")
    print(f"This is percentage error z {np.mean(z_err)}")

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot drone trajectory
    ax.plot(x_hist, y_hist, z_hist, 'b', linewidth=2, label='Drone Trajectory')

    # Plot reference trajectory
    ax.plot(x_ref, y_ref, z_ref, 'k--', linewidth=1.5, label='Reference')

    # Mark start and end points
    ax.scatter(start_x, start_y, start_z, c='g', marker='o', s=100, label='Start')
    ax.scatter(end_x, end_y, end_z, c='r', marker='x', s=100, label='End')

    # Set labels and title
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title(f'Trajectory Tracking - {trajectory_name} Trajectory')

    # Set ticks at 0.5 increments
    max_range = max(np.ptp(x_hist), np.ptp(y_hist), np.ptp(z_hist))  # Peak-to-peak range
    tick_step = 0.5
    ticks = np.arange(0, max_range + tick_step, tick_step)  # Generate ticks (0, 0.5, 1, ...)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib 3.3.0 or later

    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_singluar_vid(drone_history: np.ndarray, ref_states: np.ndarray, start_point: np.ndarray, end_point: np.ndarray, trajectory_name: str) -> Figure:
    # Extract states
    x_ref, y_ref, z_ref = ref_states[:, 0], ref_states[:, 2], ref_states[:, 4]
    x_hist, y_hist, z_hist = drone_history[:, 0], drone_history[:, 2], drone_history[:, 4]
    start_x, start_y, start_z = start_point
    end_x, end_y, end_z = end_point
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot drone trajectory
    ax.plot(x_hist, y_hist, z_hist, 'b', linewidth=2, label='Drone Trajectory')

    # Plot reference trajectory
    ax.plot(x_ref, y_ref, z_ref, 'k--', linewidth=1.5, label='Reference')

    # Mark start and end points
    ax.scatter(start_x, start_y, start_z, c='g', marker='o', s=100, label='Start')
    ax.scatter(end_x, end_y, end_z, c='r', marker='x', s=100, label='End')

    # Set labels and title
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title(f'Trajectory Tracking - {trajectory_name} Trajectory')

    # Set ticks at 0.5 increments
    max_range = 5  # Peak-to-peak range
    tick_step = 0.5
    ticks = np.arange(0, max_range + tick_step, tick_step)  # Generate ticks (0, 0.5, 1, ...)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib 3.3.0 or later

    return fig


def plot_training_metrics(episode: int, save_dir: str):
    """Plots metrics for ALL episodes within a SINGLE epoch."""
    # Load all episode JSONs for this epoch
    epoch_dir = os.path.join("src", "results", "metrics", "episode")
    if not os.path.exists(epoch_dir):
        log_print(f"No data for Epoch {episode}")
        return

    # Load and sort episode files
    episodes = []
    rewards = []
    critic_losses = []
    q_values = []
    actor_losses = []
    counter=0

    for filename in sorted(os.listdir(epoch_dir)):
        counter+=1
        if filename.endswith('.json'):
            with open(os.path.join(epoch_dir, filename)) as f:
                data = json.load(f)
                for epoch_key, metrics in data.items():
                    episodes.append(counter)  # Extract episode number
                    rewards.append(metrics["Avg Reward"])
                    critic_losses.append(metrics["Avg Critic Loss"])
                    q_values.append(metrics["Avg Q"])
                    actor_losses.append(metrics["Avg Actor Loss"])

    # Plotting (same as before, but x-axis is episodes, not epochs)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Metrics for Episode {episode}", fontsize=16)

    # Plot 1: Reward per episode
    axs[0, 0].plot(episodes, rewards, 'b-o')
    axs[0, 0].set_title("Reward per Episode")
    axs[0, 0].set_xlabel("Episode Number")
    axs[0, 0].grid(True)

    # Plot 2: Critic Loss (log scale)
    axs[0, 1].plot(episodes, critic_losses, 'r-s')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title("Critic Loss per Episode (Log Scale)")
    axs[0, 1].grid(True)

    # Plot 3: Q Values
    axs[1, 0].plot(episodes, q_values, 'g-D')
    axs[1, 0].set_title("Q Value per Episode")
    axs[1, 0].grid(True)

    # Plot 4: Actor Loss
    axs[1, 1].plot(episodes, actor_losses, 'm-^')
    axs[1, 1].set_title("Actor Loss per Episode")
    axs[1, 1].grid(True)

    plt.tight_layout()

    if save_dir:
        full_save_path = os.path.join(epoch_dir, save_dir)
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        log_print(f"Plot saved to {full_save_path}")

    plt.close()