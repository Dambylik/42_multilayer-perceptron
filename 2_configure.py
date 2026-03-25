"""
STEP 2 — MODEL ARCHITECTURE & TRAINING CONFIGURATION
"""

import os
import numpy as np
from tools.utils import section, subsection, load_session, build_layers_from_session


def main():
    session = load_session()
    if session is None:
        return

    try:
        layers = build_layers_from_session(session)
    except ValueError as err:
        print(f"  Error: {err}")
        return

    arch_str       = str(session["layer"]) if session["layer"] else "[24, 10, 8]  (default)"
    optimizer_name = "Adam" if session["adam"] else "SGD"
    lr             = session["learning_rate"]
    epochs         = session["epochs"]
    batch_size     = session["batch_size"]
    patience       = session["patience"]

    n_train = None
    if os.path.exists("generated/X_train.npy"):
        try:
            n_train = np.load("generated/X_train.npy").shape[0]
        except Exception:
            pass
    n_batches_str = f"{(n_train + batch_size - 1) // batch_size}" if n_train else "?"

    section("MODEL ARCHITECTURE & TRAINING CONFIGURATION")
    subsection("Architecture")
    print(f"  Hidden layers : {arch_str}")
    print(f"  Weight init   : He uniform\n")

    total_params = 0
    print(f"  {'#':<4}  {'Type':<10}  {'Input → Output':^18}  {'W shape':^14}  {'Params':>8}  Note")
    print(f"  {'─' * 72}")
    for i, layer in enumerate(layers):
        name = type(layer).__name__
        if name == "Dense":
            n_in, n_out = layer.weights.shape
            p           = layer.weights.size + layer.biases.size
            total_params += p
            note = "<-- output layer" if i == len(layers) - 2 else ""
            print(f"  {i:<4}  {name:<10}  {f'({n_in} → {n_out})':^18}"
                  f"  {f'({n_in}×{n_out})':^14}  {p:>8}  {note}")
        elif name == "ReLU":
            print(f"  {i:<4}  {name:<10}  {'max(0, z)':^18}  {'—':^14}  {'—':>8}")
        elif name == "Softmax":
            print(f"  {i:<4}  {name:<10}  {'exp(z)/Σexp':^18}  {'—':^14}  {'—':>8}")
    print(f"  {'─' * 72}")
    print(f"  {'Total trainable parameters':>50} : {total_params}")

    subsection("Training configuration")
    print(f"  Loss function : CategoricalCrossEntropy")
    print(f"  Optimizer     : {optimizer_name}  (lr = {lr})")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size} ")
    print(f"  Early stop    : patience = {patience} epochs")


if __name__ == "__main__":
    main()
