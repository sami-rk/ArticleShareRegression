"""
Data loading utilities for Task 2: Regression Pipeline
"""

import pandas as pd
from config import TRAIN_DATA_PATH, TEST_DATA_PATH


def load_data():
    """Load training and test datasets."""
    train = pd.read_excel(TRAIN_DATA_PATH)
    test = pd.read_excel(TEST_DATA_PATH)
    return train, test
