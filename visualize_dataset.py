import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


def visualize_data(filename):
    df = pd.read_csv(filename, header=None)

    # --- 1. Target Distribution ---
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
    plt.show()

    # --- 2. Full Correlation Heatmap ---
    features_df = df.iloc[:, 2:]
    plt.figure(figsize=(15, 12))
    correlation_matrix = features_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Full Feature Correlation Heatmap (30x30)')
    plt.savefig("images/correlation_heatmap.png", dpi=150)
    print("Saved images/correlation_heatmap.png")
    plt.show()


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
