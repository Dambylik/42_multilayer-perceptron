import numpy as np
import csv
import pandas as pd

def standardize(X, means=None, stds=None, epsilon=1e-8):
    """
    Z-score standartization
    """
    if X.size == 0:
        return X, means, stds
    if means is None:
        means = np.mean(X, axis=0)
    if stds is None:
        stds = np.std(X, axis=0)
    return (X - means) / (stds + epsilon), means, stds


def shuffle_data(X, y):
    """Shuffle features and labels together while keeping pair alignment."""
    indices = np.random.permutation(X.shape[0])
    return X[indices], y[indices]


def load_raw_dataset(csv_path):
    """Read CSV and return numeric feature matrix plus raw labels."""
    X_list = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            X_list.append(row[2:])
            labels.append(row[1])

    X = np.array([[float(x) for x in row] for row in X_list], dtype=float)
    return X, labels


def create_set_sigmoid(csv_path, means=None, stds=None):
    """
    Like create_training_set but for Sigmoid inference.
    Labels: M → [1]   B → [0]
    """
    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1] if label == "M" else [0] for label in labels], dtype=float)
    X_norm, means, stds = standardize(X, means, stds)
    return X_norm, y, means, stds


def create_training_set(csv_path, means=None, stds=None):
    """
    Load CSV, one-hot encode labels, and standardize features.
    Labels: M → [1, 0]   B → [0, 1]
    """
    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1, 0] if label == "M" else [0, 1] for label in labels], dtype=float)
    X_standart, means, stds = standardize(X, means, stds)
    return X_standart, y, means, stds


def split_dataset(csv_path, validation_ratio, train_out="train_set.csv", val_out="validation_set.csv"):
    """Split a CSV dataset into train and validation files."""
    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")
    data = pd.read_csv(csv_path, header=None)
    validation_set = data.sample(frac=validation_ratio)
    train_set = data.drop(validation_set.index)
    validation_set.to_csv(val_out, index=False, header=False)
    train_set.to_csv(train_out, index=False, header=False)
    print(f"Train set:      {len(train_set)} samples → {train_out}")
    print(f"Validation set: {len(validation_set)} samples → {val_out}")
    print("---" * 50)


