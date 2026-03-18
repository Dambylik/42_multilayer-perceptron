import sys
import math
import random
import csv

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


# --- Main Logic ---
def main():
    X, y = load_csv('data.csv')
    X, y = shuffle_data(X, y)
    
    # Split: 80% Train, 20% Validation
    split = int(len(X) * 0.8)
    train_X, val_X = X[:split], X[split:]
    train_y, val_y = y[:split], y[split:]
    
    # Standardize: Use TRAIN stats to scale BOTH sets
    means, stds = calculate_stats(train_X)
    train_X = apply_standardization(train_X, means, stds)
    val_X = apply_standardization(val_X, means, stds)

    print(f"Training set size: {len(train_X)}")
    print(f"Validation set size: {len(val_X)}")
    print(f"Training labels size: {len(train_y)}")
    print(f"Validation labels size: {len(val_y)}")
    print(f"Features size: {len(X[0])}")

if __name__ == "__main__":
    main()