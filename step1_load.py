"""
STEP 1 — LOAD & PREPROCESS DATA
"""

import json
import numpy as np
from tools.utils import create_training_set, section, subsection
from tools.visualize_dataset import print_dataset_info


def main():
    try:
        with open("generated/session.json") as f:
            session = json.load(f)
    except FileNotFoundError:
        print("  Error: session.json not found.")
        return

    train_path = session["train_path"]
    val_path   = session["val_path"]
    section("LOAD & PREPROCESS DATA")
    print(f"""
  For each CSV we perform:

  1. READ CSV
      col 0  → ID         (discarded)
      col 1  → label      M or B
      col 2-31 → 30 numeric features

  2. ONE-HOT ENCODE LABELS
      The network has 2 output neurons: [neuron_M, neuron_B].
      We encode the true label as a TARGET probability distribution:
        M (Malignant)  →  [1, 0]   "100 % malignant, 0 % benign"
        B (Benign)     →  [0, 1]   "0 % malignant, 100 % benign"

  3. Z-SCORE STANDARDISATION
      Formula: X_norm = (X - mean) / std
      After normalisation every feature has mean ≈ 0 and std ≈ 1.
""")

    subsection("Loading and preprocessing training set")
    try:
        X_train, y_train, train_means, train_stds = create_training_set(train_path)
    except FileNotFoundError as err:
        print(f"  Error: {err}")
        return

    subsection("Loading and preprocessing validation set  (reusing training stats)")
    try:
        X_val, y_val, _, _ = create_training_set(val_path, train_means, train_stds)
    except FileNotFoundError as err:
        print(f"  Error: {err}")
        return

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save("generated/X_train.npy", X_train)
    np.save("generated/y_train.npy", y_train)
    np.save("generated/X_val.npy",   X_val)
    np.save("generated/y_val.npy",   y_val)

    norm_stats = {
        "means": train_means.tolist(),
        "stds":  train_stds.tolist()
    }
    with open("generated/norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=4)


if __name__ == "__main__":
    main()
