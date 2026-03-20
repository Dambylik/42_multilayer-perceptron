import numpy as np
import csv
import pandas as pd

def normalization(X, means=None, stds=None, epsilon=1e-8):
    """
    Z-score normalization
    Args:
        X: 2D numpy array of shape (n_samples, n_features)
        means: optional per-feature means (computed from X if None)
        stds: optional per-feature stds (computed from X if None)
        epsilon: numerical stability term to avoid division by zero

    Returns:
        tuple: (X_norm, means, stds)
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
            X_list.append(row[2:])  # Skip id and target columns
            labels.append(row[1])

    X = np.array([[float(x) for x in row] for row in X_list], dtype=float)
    return X, labels


# Return dataset with labels adapted for Sigmoid (no one-hot).
# Pass means/stds computed from training data to avoid data leakage on validation sets.
def create_set_sigmoid(csv_path, means=None, stds=None):
    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1] if label == "M" else [0] for label in labels], dtype=float)
    X_norm, means, stds = normalization(X, means, stds)
    return X_norm, y, means, stds

# Return dataset with labels adapted for Softmax (one-hot).
# Pass means/stds computed from training data to avoid data leakage on validation sets.
def create_set_softmax(csv_path, means=None, stds=None):
    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1, 0] if label == "M" else [0, 1] for label in labels], dtype=float)
    X_norm, means, stds = normalization(X, means, stds)
    return X_norm, y, means, stds


def split_dataset(csv_path, validation_ratio, train_out="train_set.csv", val_out="validation_set.csv"):
    """Split a CSV dataset into train and validation files."""
    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")
    data = pd.read_csv(csv_path)
    validation_set = data.sample(frac=validation_ratio)
    train_set = data.drop(validation_set.index)
    validation_set.to_csv(val_out, index=False)
    train_set.to_csv(train_out, index=False)
    print(f"Train set:      {len(train_set)} samples → {train_out}")
    print(f"Validation set: {len(validation_set)} samples → {val_out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 utils.py <file.csv> [validation_ratio=0.2]")
        sys.exit(1)
    try:
        ratio = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.2
        split_dataset(sys.argv[1], ratio)
    except (ValueError, FileNotFoundError) as err:
        print(f"Error: {err}")
        sys.exit(1)
