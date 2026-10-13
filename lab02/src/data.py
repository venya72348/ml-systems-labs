"""Load load_digits, scale to [0, 1], one-hot labels, 70/30 stratified split."""
import random

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0                      # pixel values 0..16 -> 0..1
    Y = np.eye(10)[y]                 # one-hot labels, shape (n, 10)
    return train_test_split(X, Y, test_size=0.3, stratify=y, random_state=SEED)