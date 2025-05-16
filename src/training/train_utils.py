import numpy as np
import json
import os
import torch

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
    dataset_file_path = os.path.join(os.getcwd(), "src/data", filename)
    with open(dataset_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def create_directory_checkpoint():
    directory = os.path.join(os.getcwd(), "src", "checkpoint")
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory '{directory}'")
        else:
            print(f"Directory '{directory}' already exists")
    except PermissionError:
        print(f"Error: No permission to create '{directory}'")
    return directory


def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, epoch):
    checkpoint_path = os.path.join(create_directory_checkpoint(), f"td3_quadcopter_epoch_{epoch + 1}.pth")
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }, checkpoint_path)
