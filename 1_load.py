"""
STEP 1 — LOAD & PREPROCESS DATA
"""

import numpy as np
from tools.utils import load_dataset, section, subsection, load_session
from tools.visualize_dataset import visualize_data, visualize_training_heatmap


def main():
    session = load_session()
    if session is None:
        return

    train_path = session["train_path"]
    val_path   = session["val_path"]
    
    section("LOAD & PREPROCESS DATA")
    subsection("Loading and preprocessing training set")
    try:
        X_train, y_train, train_means, train_stds = load_dataset(train_path)
    except FileNotFoundError as err:
        print(f"  Error: {err}")
        return

    subsection("Loading and preprocessing validation set")
    try:
        X_val, y_val, _, _ = load_dataset(val_path, train_means, train_stds)
    except FileNotFoundError as err:
        print(f"  Error: {err}")
        return

    print("\n")
    visualize_data(train_path)
    visualize_training_heatmap(X_train, y_train)
    np.save("generated/X_train.npy", X_train)
    np.save("generated/y_train.npy", y_train)
    np.save("generated/X_val.npy",   X_val)
    np.save("generated/y_val.npy",   y_val)

    import json
    norm_stats = {
        "means": train_means.tolist(),
        "stds":  train_stds.tolist()
    }
    with open("generated/norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=4)


if __name__ == "__main__":
    main()
