import pickle
import torch


def load_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
