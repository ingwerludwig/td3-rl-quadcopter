import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.init_dataset import start as start_generate_dataset
from training.train_td3 import start as start_train
from training.train_utils import load_dataset
from src.config.constant import DATASET_FILENAME, DATASET_VALIDATION_FILENAME


def main():
    start_generate_dataset(DATASET_FILENAME)
    start_generate_dataset(DATASET_VALIDATION_FILENAME)
    dataset = load_dataset(DATASET_FILENAME)
    val_dataset = load_dataset(DATASET_VALIDATION_FILENAME)
    start_train(dataset, val_dataset)

if __name__ == "__main__":
    main()