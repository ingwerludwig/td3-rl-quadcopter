import sys
from typing import List, Dict
import json
from .init_dataset_utils import generate_hover, generate_line, generate_circle, generate_s_curve, generate_step
from src.config.constant import NUM_EPISODE
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

def generate_dataset():
    dataset = []
    for i in range(NUM_EPISODE):
        dataset.append(generate_hover())
    return dataset


def save_to_json(dataset: List[Dict], filename):
    """Save dataset to JSON file."""
    os.makedirs(os.path.join(os.getcwd(), "src/data"), exist_ok=True)
    dataset_file_path = os.path.join(os.getcwd(), "src/data", filename)
    with open(dataset_file_path, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def start(filename):
    dataset = generate_dataset()
    save_to_json(dataset, filename)

    print(f"Generated {len(dataset)} trajectories:")
    print(f"Saved to {filename}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_dataset.py <output_filename.json>")
    else:
        start(sys.argv[1])