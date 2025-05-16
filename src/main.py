from data.init_dataset import start as start_generate_dataset
from training.train_td3 import start as start_train
from training.train_utils import load_dataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    filename = "lqr_trajectories.json"
    start_generate_dataset(filename)
    dataset = load_dataset(filename)
    start_train(dataset)

if __name__ == "__main__":
    main()