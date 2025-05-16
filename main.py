from init_dataset import start as start_generate_dataset
from train import start as start_train
from train_utils import load_dataset
import pandas as pd
import json
import os

def main():
    filename = "lqr_trajectories.json"
    start_generate_dataset(filename)
    dataset = load_dataset(filename)
    start_train(dataset)

if __name__ == "__main__":
    main()