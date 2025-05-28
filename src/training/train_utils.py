import json
import os
import torch
import shutil
from typing import Dict, Any, Optional, List
from src.agents.td3 import TD3


def load_dataset(filename):
    dataset_file_path = os.path.join(os.getcwd(), "src/data", filename)
    with open(dataset_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset


def create_directory_checkpoint():
    directory = os.path.join(os.getcwd(), "src", "checkpoint")
    os.makedirs(directory, exist_ok=True)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory '{directory}'")
    except PermissionError:
        print(f"Error: No permission to create '{directory}'")
    return directory


def save_checkpoint(agent: TD3, epoch):
    checkpoint_path = os.path.join(create_directory_checkpoint(), f"td3_quadcopter_epoch_{epoch + 1}.pth")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic1.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, checkpoint_path)


def save_metrics_to_json(
        metrics: Dict[str, Any],
        filepath: str
):

    base_dir = os.path.join(os.getcwd(), "src", "results", "metrics", "episode")
    full_path = os.path.join(base_dir, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        os.remove(full_path)

    with open(full_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return None


def delete_existing(
        subdir_names: Optional[List[str]] = None,
        parent_dir: str = os.path.join("src")
):

    parent_dir = os.path.join(os.getcwd(), parent_dir)
    if not os.path.exists(parent_dir):
        return None

    deleted_dirs = []

    if subdir_names:
        for subdir in subdir_names:
            dir_path = os.path.join(parent_dir, subdir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                deleted_dirs.append(dir_path)
    else:
        raise ValueError("Either `subdir_names` or `pattern` must be provided.")

def create_metrics_episode(ep_i, avg_reward, avg_critic_loss, avg_q_value, avg_actor_loss):
    return  {
            f"Episode {ep_i + 1}": {
                "Avg Reward": float(round(avg_reward, 2)),
                "Avg Critic Loss": float(round(avg_critic_loss, 2)),
                "Avg Q": float(round(avg_q_value, 2)),
                "Avg Actor Loss": float(round(avg_actor_loss, 2))
            }
        }


def log_print(*args, **kwargs):
    logs_dir = os.path.join(os.getcwd(), "src", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_filepath = os.path.join(logs_dir, "result.log")
    os.makedirs(logs_dir, exist_ok=True)
    with open(log_filepath, 'a') as f:
        print(*args, **kwargs, file=f)