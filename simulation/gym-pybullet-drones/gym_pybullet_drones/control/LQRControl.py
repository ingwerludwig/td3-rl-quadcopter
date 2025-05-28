import gymnasium as gym
import numpy as np
from control import lqr, ctrb
from gymnasium import spaces

from gym_pybullet_drones.control.quadcopter import Quadcopter
from src.controllers.lqr_quadcopter_controller import LQRQuadcopterController


class QuadcopterLQREnv(gym.Env):
    def __init__(self, quadcopter: Quadcopter, render_mode=None):
        super(QuadcopterLQREnv, self).__init__()

        # Q and R actions for LQR Controller
        self.action_space = spaces.Box(
            low=0.01, high=np.inf, shape=(16,)
        )

        # Observation space: state variables
        # [x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,)
        )

        # Physics parameters
        self.dt = 0.01
        self.gravity = 9.81
        self.quadcopter_mass = quadcopter.mass
        self.Ixx = quadcopter.Ixx
        self.Iyy = quadcopter.Iyy
        self.Izz = quadcopter.Izz

        # Quadcopter control settings
        self.T_hover = quadcopter.mass * self.gravity
        self.T_max = quadcopter.T_max
        self.T_min = quadcopter.T_min
        self.torque_max = quadcopter.torque_max
        self.torque_min = quadcopter.torque_min

        # Controller initialization
        self.lqr_controller = LQRQuadcopterController(quadcopter_config=quadcopter)

        # Initialize state
        self.states = None
        self.ref_states = None
        self.quadcopter_actions = np.array([self.T_hover, 0, 0, 0])
        self.render_mode = render_mode

    def reset(self, ref_states, seed=None, options=None):
        super().reset(seed=seed)
        self.states = np.zeros(12)
        self.ref_states = ref_states
        return self.states, {}

    def step(self):
        """Move the quadcopter based on the quadcopter action for dt step"""
        # Apply physics model to the quadcopter
        self._apply_dynamics()

        # Calculate reward
        reward = self._calculate_reward()


        # Check termination conditions
        terminated = self._check_termination()

        # Truncation (if using time limit)
        truncated = False

        # Additional info
        info = {"status": "Moving :) ........"}

        if terminated:
            reward += -100
            info['status'] = "Crashed :( ......."

        if abs(reward) <= 0.0001:
            terminated = True
            info['status'] = "Reached the reference state"

        return self.states, reward, terminated, truncated, info

    def set_ref_states(self, ref_states):
        self.ref_states = ref_states

    def _apply_dynamics(self):
        """Update states using RK4 integration for better accuracy."""
        def dynamics(states, actions):
            x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = states
            T, tau_phi, tau_theta, tau_psi = actions

            # Small-angle approximation (near-hover)
            ax = -theta * T / self.quadcopter_mass
            ay = phi * T / self.quadcopter_mass
            az = T / self.quadcopter_mass - self.gravity

            # Angular accelerations
            phi_dot_dot = tau_phi / self.Ixx
            theta_dot_dot = tau_theta / self.Iyy
            psi_dot_dot = tau_psi / self.Izz

            return np.array([
                x_dot, ax, y_dot, ay, z_dot, az,
                phi_dot, phi_dot_dot, theta_dot, theta_dot_dot,
                psi_dot, psi_dot_dot
            ])

        # RK4 Integration
        k1 = dynamics(self.states, self.quadcopter_actions)
        k2 = dynamics(self.states + 0.5 * self.dt * k1, self.quadcopter_actions)
        k3 = dynamics(self.states + 0.5 * self.dt * k2, self.quadcopter_actions)
        k4 = dynamics(self.states + self.dt * k3, self.quadcopter_actions)
        self.states += (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Wrap angles to [-π, π] for stability
        self.states[6] = np.arctan2(np.sin(self.states[6]), np.cos(self.states[6]))  # phi
        self.states[8] = np.arctan2(np.sin(self.states[8]), np.cos(self.states[8]))  # theta
        self.states[10] = np.arctan2(np.sin(self.states[10]), np.cos(self.states[10]))  # psi

    def compute_quad_actions_from_controller(self, states=None, ref_states=None):
        """states and ref_states should be provided for external calculations"""
        states_var = states if states is not None else self.states
        ref_states_var = ref_states if ref_states is not None else self.ref_states

        dthrust, p, q, r = self.lqr_controller.get_actions(states_var, ref_states_var)

        # clip actions
        total_thrust = np.clip(self.quadcopter_mass * self.gravity + dthrust, self.T_min, self.T_max)

        phi_torque = np.clip(p, self.torque_min, self.torque_max)
        theta_torque = np.clip(q, self.torque_min, self.torque_max)
        psi_torque = np.clip(r, self.torque_min, self.torque_max)

        self.states = states_var
        self.ref_states = ref_states_var
        self.quadcopter_actions = np.array([total_thrust, phi_torque, theta_torque, psi_torque])

        return np.array([total_thrust, phi_torque, theta_torque, psi_torque])

    def update_lqr_params(self, lqr_params):
        """Update LQR parameters"""
        new_Q = np.diag(lqr_params[:12])
        new_R = np.diag(lqr_params[12:])

        self.lqr_controller.set_Q(new_Q)
        self.lqr_controller.set_R(new_R)

    def _convert_to_rpm(self, actions):
        """
        Convert desired actions (thrust, roll, pitch, yaw) into motor RPMs.

        Args:
            actions: [T, p, q, r]

        Returns:
            list: RPMs for motors [M1, M2, M3, M4].
        """
        k = 2.3544e-9  # N/RPM^2
        T, p, q, r = actions
        max_rpm = 15000

        # Compute motor forces (F = k * RPM²)
        F1 = (T + p + q - r) / 4
        F2 = (T - p + q + r) / 4
        F3 = (T - p - q - r) / 4
        F4 = (T + p - q + r) / 4

        # Convert forces to RPMs (RPM = sqrt(F / k))
        rpm1 = np.sqrt(F1 / k)
        rpm2 = np.sqrt(F2 / k)
        rpm3 = np.sqrt(F3 / k)
        rpm4 = np.sqrt(F4 / k)

        # Clip RPMs to avoid exceeding motor limits
        rpm1 = np.clip(rpm1, 0, max_rpm)
        rpm2 = np.clip(rpm2, 0, max_rpm)
        rpm3 = np.clip(rpm3, 0, max_rpm)
        rpm4 = np.clip(rpm4, 0, max_rpm)

        return [rpm1, rpm2, rpm3, rpm4]

    def _calculate_reward(self) -> float:
        # Calculate error only on position and velocity, focus x,y,z and angle
        state_error = self.states-self.ref_states
        position_error = np.sum(np.array([state_error[0], state_error[2], state_error[4]])**2)
        stability_error = np.sum(np.array([state_error[6], state_error[8], state_error[10]])**2)
        reward = -(position_error + 0.5 * stability_error)
        return reward

    def _check_termination(self) -> bool:
        # Terminate if crashed or flipped
        z = self.states[4]

        phi, theta = abs(self.states[6]), abs(self.states[8])

        crashed = z <= 0
        flipped = phi > np.pi/2 or theta > np.pi/2
        return crashed or flipped