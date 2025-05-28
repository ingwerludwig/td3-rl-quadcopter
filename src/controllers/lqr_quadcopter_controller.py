import numpy as np
from control import lqr, ctrb
from src.models.dynamics.quadcopter import Quadcopter
from src.training.train_utils import log_print


class LQRQuadcopterController:
    def __init__(self, quadcopter_config: Quadcopter):
        gravity = 9.81 # kg.m/s^2

        # State matrix A (linearized around hover)
        self.A = np.zeros((12, 12))
        self.A[0, 1] = 1  # x_dot = vx
        self.A[2, 3] = 1  # y_dot = vy
        self.A[4, 5] = 1  # z_dot = vz
        self.A[1, 8] = -gravity  # vx_dot = -g * theta
        self.A[3, 6] = gravity   # vy_dot = g * phi
        self.A[6, 7] = 1   # phi_dot = p
        self.A[8, 9] = 1   # theta_dot = q
        self.A[10, 11] = 1 # psi_dot = r

        # Input matrix B (inputs: [delta_T, tau_phi, tau_theta, tau_psi])
        self.B = np.zeros((12, 4))
        self.B[5, 0] = 1 / quadcopter_config.mass   # delta_T -> vz_dot
        self.B[7, 1] = 1 / quadcopter_config.Ixx  # tau_phi -> p_dot
        self.B[9, 2] = 1 / quadcopter_config.Iyy  # tau_theta -> q_dot
        self.B[11, 3] = 1 / quadcopter_config.Izz # tau_psi -> r_dot

        # Cost matrices
        self.Q = np.diag([10, 1, 10, 1, 10, 1, 5, 0.5, 5, 0.5, 1, 0.1])
        self.R = np.diag([1, 10, 10, 10])

        # Check controllability
        controllability_matrix = ctrb(self.A, self.B)
        rank = np.linalg.matrix_rank(controllability_matrix)
        if rank != 12:
            print(f"Warning: System is not controllable. Rank: {rank}/12")

        # Compute LQR gain
        self.K = self._compute_gain()

    def _compute_gain(self):
        try:
            K, _, _ = lqr(self.A, self.B, self.Q, self.R)
            return K
        except Exception as e:
            print(f"LQR computation failed: {e}")
            raise

    def get_actions(self, curr_states, ref_states):
        state_error = curr_states - ref_states
        actions = -np.dot(self.K, state_error)  # Outputs [delta_T, tau_phi, tau_theta, tau_psi]
        return actions

    def set_Q(self, Q):
        self.Q = Q

    def set_R(self, R):
        self.R = R