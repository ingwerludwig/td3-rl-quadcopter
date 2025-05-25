import copy
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.integrate import odeint

from src.envs.quadcopter_lqr_env import QuadcopterLQREnv
from src.models.dynamics.quadcopter import Quadcopter
from src.training.train_utils import log_print


def update_states(self, dt):
        """
        Update the quadcopter states based on current states and actions.
        dt: time step
        """
        # Extract states (already in correct format)
        x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = self.states

        # Extract actions
        T, tau_phi, tau_theta, tau_psi = self.actions

        # Gravity
        g = self.g

        # Rotation matrix components
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # Linear accelerations in inertial frame
        ax = -(stheta) * T / self.m
        ay = (sphi * ctheta) * T / self.m
        az = (cphi * ctheta) * T / self.m - g

        # Angular accelerations
        phi_dot_dot = tau_phi / self.Ixx
        theta_dot_dot = tau_theta / self.Iyy
        psi_dot_dot = tau_psi / self.Izz

        # Update states using Euler integration
        # Position updates
        new_x = x + x_dot * dt
        new_y = y + y_dot * dt
        new_z = z + z_dot * dt

        # Velocity updates
        new_x_dot = x_dot + ax * dt
        new_y_dot = y_dot + ay * dt
        new_z_dot = z_dot + az * dt

        # Angle updates
        new_phi = phi + phi_dot * dt
        new_theta = theta + theta_dot * dt
        new_psi = psi + psi_dot * dt

        # Angular rate updates
        new_phi_dot = phi_dot + phi_dot_dot * dt
        new_theta_dot = theta_dot + theta_dot_dot * dt
        new_psi_dot = psi_dot + psi_dot_dot * dt

        # Update state vector (maintaining format)
        self.states = np.array([
            new_x, new_x_dot,  # x, x_dot
            new_y, new_y_dot,  # y, y_dot
            new_z, new_z_dot,  # z, z_dot
            new_phi, new_phi_dot,  # phi, phi_dot
            new_theta, new_theta_dot,  # theta, theta_dot
            new_psi, new_psi_dot  # psi, psi_dot
        ])

def quadcopter_dynamics(state, t, u_func, quad: Quadcopter):
    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi = state
    m, Ixx, Iyy, Izz, g = quad.mass, quad.Ixx, quad.Iyy, quad.Izz, quad.g

    # Control input
    U1, U2, U3, U4 = u_func(t, state)

    # Rotation matrix components
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # Translational dynamics
    ddx = -(U1 / m) * (cphi * stheta * cpsi + sphi * spsi)
    ddy = -(U1 / m) * (cphi * stheta * spsi - sphi * cpsi)
    ddz = -g + (U1 / m) * cphi * ctheta

    # Rotational kinematics
    dphi_dt = dphi + dtheta * sphi * np.tan(theta) + dpsi * cphi * np.tan(theta)
    dtheta_dt = dtheta * cphi - dpsi * sphi
    dpsi_dt = dtheta * sphi / ctheta + dpsi * cphi / ctheta

    # Rotational dynamics
    ddphi = (U2 / Ixx) + ((Iyy - Izz) / Ixx) * dtheta * dpsi
    ddtheta = (U3 / Iyy) + ((Izz - Ixx) / Iyy) * dphi * dpsi
    ddpsi = (U4 / Izz) + ((Ixx - Iyy) / Izz) * dphi * dtheta

    return [dx, ddx, dy, ddy, dz, ddz, dphi_dt, ddphi, dtheta_dt, ddtheta, dpsi_dt, ddpsi]


def simulate_quadcopter(lqr_quad_env:QuadcopterLQREnv, quad: Quadcopter, initial_state, ref_state, t_total=40):
    dt=0.01
    lqr_quad_env_copy = copy.deepcopy(lqr_quad_env)
    controller_copy = copy.deepcopy(lqr_quad_env_copy.lqr_controller)
    t_vec = np.arange(0, t_total, dt)

    # Create a constant reference trajectory
    ref_states = np.array([ref_state]).T * np.ones((12, len(t_vec)))
    ref_acc = np.zeros((6, len(t_vec)))  # Zero acceleration references

    def u_func(t, state):
        lqr_quad_env_copy.states = state
        t_idx = np.argmin(np.abs(t_vec - t))
        current_ref = ref_states[:, t_idx]
        return controller_copy.get_actions(initial_state, ref_states[:, t_idx])

    # Simulate
    sol = odeint(quadcopter_dynamics, initial_state, t_vec, args=(u_func, quad))

    # Compute control inputs
    u = np.array([u_func(t, sol[i]) for i, t in enumerate(t_vec)])

    return t_vec, sol, u


def plot_results(t_vec, sol, ref_state, u, episode, sample, filename):
    x, y, z = sol[:, 0], sol[:, 2], sol[:, 4]
    phi, theta, psi = sol[:, 6], sol[:, 8], sol[:, 10]

    # Extract reference states
    x_ref, y_ref, z_ref = ref_state[0], ref_state[2], ref_state[4]
    phi_ref, theta_ref, psi_ref = ref_state[6], ref_state[8], ref_state[10]

    plt.figure(figsize=(12, 8))

    # 3D Trajectory
    ax = plt.subplot(2, 2, 1, projection='3d')
    ax.plot(x, y, z, label='Actual')
    ax.scatter(x_ref, y_ref, z_ref, c='r', marker='x', s=100, label='Reference')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()

    # Position Errors
    plt.subplot(2, 2, 2)
    plt.plot(t_vec, x - x_ref, label='X Error')
    plt.plot(t_vec, y - y_ref, label='Y Error')
    plt.plot(t_vec, z - z_ref, label='Z Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid()

    # Attitude
    plt.subplot(2, 2, 3)
    plt.plot(t_vec, np.degrees(phi), label='φ Actual')
    plt.plot(t_vec, np.degrees(theta), label='θ Actual')
    plt.plot(t_vec, np.degrees(psi), label='ψ Actual')
    plt.axhline(np.degrees(phi_ref), linestyle='--', color='C0', label='φ Ref')
    plt.axhline(np.degrees(theta_ref), linestyle='--', color='C1', label='θ Ref')
    plt.axhline(np.degrees(psi_ref), linestyle='--', color='C2', label='ψ Ref')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid()

    # Control Inputs
    plt.subplot(2, 2, 4)
    plt.plot(t_vec, u[:, 0], label='U1 (N)')
    plt.plot(t_vec, u[:, 1], label='U2 (Nm)')
    plt.plot(t_vec, u[:, 2], label='U3 (Nm)')
    plt.plot(t_vec, u[:, 3], label='U4 (Nm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Control')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    save_dirpath = os.path.join(os.getcwd(), "src", "results", "visualization")
    os.makedirs(save_dirpath, exist_ok=True)

    save_path = os.path.join(save_dirpath, f"Episode_{episode}_sample_{sample}_{filename}")
    plt.savefig(save_path)  # Save the plot to the specified path
    plt.close()  # Close the figure to free memory

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