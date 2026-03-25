import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np


def print_dataset_info(name, X, y):
    """Print shape, class counts, first sample, and a class distribution bar chart."""
    np.set_printoptions(suppress=True, precision=4)
    print(f"{name} shape : {X.shape}")
    print(f"  Malignant (M) : {int(np.sum(y[:, 0]))}")
    print(f"  Benign    (B) : {int(np.sum(y[:, 1]))}")
    print(f"  First sample  : {X[0]}")
    print("---" * 50)

    # counts = [int(np.sum(y[:, 1])), int(np.sum(y[:, 0]))]
    # fig, ax = plt.subplots(figsize=(5, 4))
    # bars = ax.bar(['Benign (B)', 'Malignant (M)'], counts, color=['steelblue', 'tomato'])
    # ax.bar_label(bars)
    # ax.set_title(f'Class Distribution — {name}')
    # ax.set_ylabel('Number of samples')
    # plt.tight_layout()
    # plt.show()


def visualize_data(filename):
    df = pd.read_csv(filename, header=None)
    counts = df[1].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["steelblue", "tomato"])
    plt.title("Class Distribution (Malignant vs Benign)")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        plt.text(i, val + 1, str(val), ha="center")
    plt.tight_layout()
    plt.savefig("images/class_distribution.png", dpi=150)
    print("Saved images/class_distribution.png")
    
    features_df = df.iloc[:, 2:]
    plt.figure(figsize=(15, 12))
    correlation_matrix = features_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Full Feature Correlation Heatmap (30x30)')
    plt.savefig("images/correlation_heatmap.png", dpi=150)
    print("Saved images/correlation_heatmap.png")
    


def visualize_training_heatmap(X_train, y_train):
    """Plot and save a correlation heatmap of the normalized training features."""
    df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Training Set Feature Correlation Heatmap (normalized)')
    plt.tight_layout()
    plt.savefig("images/training_heatmap.png", dpi=150)
    print("Saved images/training_heatmap.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset before training")
    parser.add_argument("file", nargs="?", default="data.csv", help="Dataset CSV file")
    args = parser.parse_args()

    try:
        visualize_data(args.file)
    except FileNotFoundError:
        print(f"Error: file '{args.file}' not found")


if __name__ == "__main__":
    main()
