import sys
import math
import random
import csv
import numpy as np

# =====================
# Data Loading
# =====================

def load_csv(filename):
    X, y = [], []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            # enumerate returns (index, value)
            for line_num, row in enumerate(reader, 1):
                try:
                    # Label is at index 1: 'M' -> 1, 'B' -> 0
                    label = 1 if row[1] == 'M' else 0
                    # Features start from index 2 to the end
                    features = [float(val) for val in row[2:]]
                    X.append(features)
                    y.append(label)
                except (ValueError, IndexError):
                    print(f"Error parsing line {line_num}. Skipping.")
    except FileNotFoundError:
        print(f"File {filename} not found."); sys.exit(1)
    return X, y


# =====================
# Data shuffling
# =====================

def shuffle_data(X, y):
    """
    Shuffle X and y together using Fisher-Yates algorithm.
    X: list of feature vectors
    y: list of labels
    Returns: shuffled X and y (as tuple)
    """
    random.seed(42)
    indices = list(range(len(X)))
    for i in range(len(indices) - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    
    # Logic: Reorder both lists using the same shuffled indices
    X_shuffled = [X[i] for i in indices]
    y_shuffled = [y[i] for i in indices]
    return X_shuffled, y_shuffled

# =====================
# Standartization
# =====================

def calculate_stats(dataset):
    # Calculate Mean and Std Dev for each of the 30 features
    num_features = len(dataset[0])
    means, stds = [], []
    for i in range(num_features):
        col = [row[i] for row in dataset]
        mean = sum(col) / len(col)
        # Variance calculation
        var = sum((x - mean)**2 for x in col) / len(col)
        means.append(mean)
        stds.append(math.sqrt(var) + 1e-8) # Add epsilon to avoid /0
    return means, stds


def apply_standardization(dataset, means, stds):
    """
    Normalize using z-score normalization.
    """
    return [[(row[i] - means[i]) / stds[i] for i in range(len(row))] for row in dataset]


def split_dataset(fiename, train_file)