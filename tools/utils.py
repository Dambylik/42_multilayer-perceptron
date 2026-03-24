import numpy as np
import csv
import pandas as pd


# ─── Print helpers (imported by all other modules) ────────────────────────────

def section(title):
    print(f"\n{'═'*70}\n  {title}\n{'═'*70}")

def subsection(title):
    print(f"\n  {'─'*60}\n  ▶  {title}\n  {'─'*60}")

# ──────────────────────────────────────────────────────────────────────────────


def standardize(X, means=None, stds=None, epsilon=1e-8):
    """
    Z-score standardization:  X_norm = (X - mean) / std
    """
    if X.size == 0:
        return X, means, stds
    if means is None:
        means = np.mean(X, axis=0)
    if stds is None:
        stds = np.std(X, axis=0)
    X_norm = (X - means) / (stds + epsilon)
    print(f"    Normed vals: min={X_norm.min():.4f}  max={X_norm.max():.4f}  mean={X_norm.mean():.4f}")
    return X_norm, means, stds


def shuffle_data(X, y):
    """Shuffle features and labels together while keeping pair alignment."""
    indices = np.random.permutation(X.shape[0])
    return X[indices], y[indices]


def load_raw_dataset(csv_path):
    """
    Read CSV and return:
      X      — float feature matrix  (n_samples, 30)
      labels — list of raw strings   ['M', 'B', ...]
    """
    print(f"\n    Loading file : {csv_path}")

    X_list = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            X_list.append(row[2:])     # cols 2-31 = features
            labels.append(row[1])      # col 1 = label

    X = np.array([[float(x) for x in row] for row in X_list], dtype=float)

    n_M = labels.count("M")
    n_B = labels.count("B")
    print(f"    Rows read    : {len(labels)}")
    print(f"    Features     : {X.shape[1]}  (columns 3 → {X.shape[1]+2} of the CSV)")
    print(f"    Malignant (M): {n_M}   Benign (B): {n_B}")
    print(f"    Feature range: min={X.min():.4f}  max={X.max():.4f}")

    return X, labels


def create_set_sigmoid(csv_path, means=None, stds=None):
    """
    Load CSV for Sigmoid inference.
    Labels: M → [[1]]   B → [[0]]   (single output neuron)
    """
    print(f"\n    Label encoding (Sigmoid / inference) :")
    print(f"      M (Malignant) → [[1]]   B (Benign) → [[0]]")
    print(f"      Reason: Sigmoid outputs a single P(Malignant) ∈ (0,1)")

    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1] if label == "M" else [0] for label in labels], dtype=float)
    X_norm, means, stds = standardize(X, means, stds)

    return X_norm, y, means, stds


def create_training_set(csv_path, means=None, stds=None):
    """
    Load CSV, one-hot encode labels, and standardize features.
    """
    X, labels = load_raw_dataset(csv_path)
    y = np.array([[1, 0] if label == "M" else [0, 1] for label in labels], dtype=float)
    X_norm, means, stds = standardize(X, means, stds)
    return X_norm, y, means, stds


def split_dataset(csv_path, validation_ratio,
                  train_out="generated/train_set.csv", val_out="generated/validation_set.csv"):
    """
    Randomly split a CSV into train and validation files.
    """
    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")

    data = pd.read_csv(csv_path, header=None)
    total = len(data)

    print(f"    Source file      : {csv_path}")
    print(f"    Total samples    : {total}")
    print(f"    Split ratio      : {int((1-validation_ratio)*100)}% train  /  "
          f"{int(validation_ratio*100)}% validation")

    validation_set = data.sample(frac=validation_ratio)
    train_set      = data.drop(validation_set.index)
    validation_set.to_csv(val_out,   index=False, header=False)
    train_set.to_csv(train_out, index=False, header=False)
