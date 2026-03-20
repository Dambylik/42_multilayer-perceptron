import argparse
import json
import sys
from utils import load_csv, shuffle_data, calculate_stats, apply_standardization


# =====================
# Main logic
# =====================
def main():
    X, y = load_csv('data.csv')
    X, y = shuffle_data(X, y)
    
    # Split: 80% Train, 20% Validation
    split = int(len(X) * 0.8)
    train_X, val_X = X[:split], X[split:]
    train_y, val_y = y[:split], y[split:]
    
    # Standardize
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