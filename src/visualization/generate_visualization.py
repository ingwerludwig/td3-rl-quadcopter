from visualization_utils import *

def generate():
    initial_state = initial_states[4]

    # Reference state (hover at z=4)
    ref_state = reference_state_5

    # Simulate for 40 seconds
    t_vec, sol, u = simulate_quadcopter(initial_state, ref_state, t_total=40)

    # Plot results
    plot_results(t_vec, sol, ref_state, u)