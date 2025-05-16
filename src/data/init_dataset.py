import sys
from typing import List, Dict
import json
from .init_dataset_utils import generate_hover, generate_line, generate_circle, generate_s_curve, generate_step
import os

def generate_lqr_dataset() -> List[Dict]:
    """Generate dataset with all trajectory types using random parameters"""
    return [
        generate_hover(),
        generate_line(),
        generate_circle(),
        generate_s_curve(),
        generate_step()
    ]

def generate_dataset() -> list[list[dict]]:
    num_episodes_per_traj = 20
    dataset = []

    for i in range(num_episodes_per_traj):
        dataset.append(generate_lqr_dataset())
    return dataset


def save_to_json(dataset: List[Dict], filename):
    """Save dataset to JSON file."""
    dataset_file_path = os.path.join(os.getcwd(), "src/data", filename)
    with open(dataset_file_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def start(filename):
    dataset = generate_dataset()
    save_to_json(dataset, filename)

    print(f"Generated {len(dataset)} trajectories:")
    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_dataset.py <output_filename.json>")
    else:
        start(sys.argv[1])