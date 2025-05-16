import numpy as np
from typing import List, Dict
import json
from init_dataset_utils import generate_hover, generate_line, generate_circle, generate_s_curve, generate_step  


def generate_lqr_dataset() -> List[Dict]:
    """Generate dataset with all trajectory types using random parameters"""
    return [
        generate_hover(),
        generate_line(),
        generate_circle(),
        generate_s_curve(),
        generate_step()
    ]

def generate_dataset() -> List[Dict]:
    num_episodes_per_traj = 20
    dataset = []

    for i in range(num_episodes_per_traj):
        dataset.append(generate_lqr_dataset())
    return dataset


def save_to_json(dataset: List[Dict], filename):
    """Save dataset to JSON file."""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def start(filename):
    dataset = generate_dataset()
    save_to_json(dataset, filename)

    print(f"Generated {len(dataset)} trajectories:")
    print(f"\nSaved to {filename}")
