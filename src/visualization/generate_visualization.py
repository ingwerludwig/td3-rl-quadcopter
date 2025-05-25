from src.visualization.visualization_utils import *

def generate_quad_sim(lqr_quad_env: QuadcopterLQREnv, quadcopter: Quadcopter, initial_state, reference_state, t_total, episode, sample, save_filename):
    t_vec, sol, u = simulate_quadcopter(lqr_quad_env, quadcopter, initial_state, reference_state, t_total=t_total)
    plot_results(t_vec, sol, reference_state, u, episode, sample, save_filename)