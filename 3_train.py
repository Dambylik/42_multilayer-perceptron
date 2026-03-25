"""
STEP 3 — TRAINING & SAVE MODEL
"""

import json
import numpy as np
import sys
from src.neural_network import NeuralNetMLP
from src.loss_functions import CategoricalCrossEntropy
from tools.utils import section, subsection, load_session, build_layers_from_session, build_optimizer


def main():
    session = load_session()
    if session is None:
        return

    required = ["generated/X_train.npy", "generated/y_train.npy",
                "generated/X_val.npy",   "generated/y_val.npy"]
    for fname in required:
        if not __import__("os").path.exists(fname):
            print(f"  Error: {fname} not found.")
            return

    X_train = np.load("generated/X_train.npy")
    y_train = np.load("generated/y_train.npy")
    X_val   = np.load("generated/X_val.npy")
    y_val   = np.load("generated/y_val.npy")

    try:
        model = NeuralNetMLP(build_layers_from_session(session))
    except ValueError as err:
        print(f"  Error: {err}")
        sys.exit(1)

    model.configure_training(CategoricalCrossEntropy(), build_optimizer(session, model.layers))

    section("TRAINING LOOP")
    model.execute_training(
        X_train, y_train,
        X_val, y_val,
        epochs=session["epochs"],
        batch_size=session["batch_size"],
        early_stopping_patience=session["patience"]
    )

    section("SAVED MODEL in generated/export.json")
    export = []
    for layer in model.layers:
        name = type(layer).__name__
        if name == "Dense":
            export.append({
                "type": "Dense",
                "W":    layer.weights.tolist(),
                "b":    layer.biases.tolist()
            })
        else:
            export.append({"type": name})

    with open("generated/export.json", "w") as f:
        json.dump(export, f, indent=4)


if __name__ == "__main__":
    main()
