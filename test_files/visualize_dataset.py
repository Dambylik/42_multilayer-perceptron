import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(filename):
    df = pd.read_csv(filename, header=None)

    # --- 1. Target Distribution ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[1], palette='magma')
    plt.title('Class Distribution (Malignant vs Benign)')
    plt.xlabel('Diagnosis (Column 1)')
    plt.ylabel('Count')
    plt.show()

    # --- 2. Full Correlation Heatmap ---
    features_df = df.iloc[:, 2:]
    plt.figure(figsize=(15, 12))
    correlation_matrix = features_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Full Feature Correlation Heatmap (30x30)')
    plt.show()

visualize_data('data.csv')