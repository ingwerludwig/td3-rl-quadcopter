import numpy as np


class Quadcopter:
    def __init__(self, m, Ixx, Iyy, Izz, g):
        self.m = m          # Mass (kg)
        self.Ixx = Ixx      # Moment of inertia about x-axis (kg·m²)
        self.Iyy = Iyy      # Moment of inertia about y-axis (kg·m²)
        self.Izz = Izz      # Moment of inertia about z-axis (kg·m²)
        self.g = g          # Gravity (m/s²)

        # Thrust and torque limits (physical constraints)
        self.T_hover = m * g  # Hover thrust (equilibrium point)
        self.T_max = m * g * 2.0  # Max thrust (2x hover)
        self.T_min = m * g * 0.05  # Min thrust (5% hover)
        self.torque_max = 0.2 * Ixx  # Max torque (scaled by inertia)
        self.torque_min = -self.torque_max

        # Initialize 12 states: [x,x_dot,y,y_dot,z,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot]
        self.states = np.zeros(12)
        self.actions = np.array([self.T_hover, 0, 0, 0])  # Start at hover


    def update_states(self, dt=0.01):
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


    def update_actions(self, T, tau_phi, tau_theta, tau_psi):
        """Update control inputs with clipping."""
        self.actions = np.clip(
            [T + self.T_hover, tau_phi, tau_theta, tau_psi],  # Add hover thrust offset
            [self.T_min, self.torque_min, self.torque_min, self.torque_min],
            [self.T_max, self.torque_max, self.torque_max, self.torque_max]
        )


    def reset(self):
        """Reset to hover condition."""
        self.states = np.zeros(12)
        self.actions = np.array([self.T_hover, 0, 0, 0])