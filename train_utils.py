import numpy as np
import json

def compute_next_states(qr_pair, current_state, s_reference_final, quad, controller):
    dt = 0.01

    quad.states = current_state
    controller.set_Q(qr_pair[0:12])
    controller.set_R(qr_pair[12:16])

    ds = (quad.states-s_reference_final)
    parameter = qr_pair
    actions = controller.get_actions(s_reference_final)

    reward = compute_reward(quad.states, actions, s_reference_final)
    quad.update_actions(actions[0], actions[1], actions[2], actions[3])
    quad.update_states(dt)

    s_next = quad.states
    s_next[4] = max(s_next[4], 0)
    ds_next = (s_next-s_reference_final)

    return ds, reward, ds_next, s_next


def compute_reward(states, actions, ref_states, w1=1.0, w2=0.1):

    # Calculate error only on position and velocity
    state_error = np.sum((states-ref_states)**2) # ||e_t||_2^2

    # Control effort penalty
    control_effort = np.sum(actions**2)  # ||u_t||_2^2

    # Combine into reward (negative cost)
    reward = -(w1 * state_error + w2 * control_effort)

    return max(reward, -1000000)

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset