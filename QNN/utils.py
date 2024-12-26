import random

def normalize_data(data):
    """
    Returns normalized data between 0 and 1.
    """
    minimum = min(data)
    maximum = max(data)
    return [(x - minimum) / (maximum - minimum) for x in data] if maximum != minimum else [0] * len(data)

def split_data(data, labels, ratio=0.8):
    """
    Splits data into training and testing sets.
    """
    split_idx = int(len(data) * ratio)
    return (data[:split_idx], labels[:split_idx]), (data[split_idx:], labels[split_idx:])

def generate_random_params(size):
    """
    Generates a list of random parameters.
    """
    return [random.random() for _ in range(size)]

def accuracy(predictions, targets):
    """
    Computes accuracy percentage.
    """
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets) if targets else 0