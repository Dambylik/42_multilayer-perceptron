"""
STEP 4 — TRAINING & SAVE MODEL

Next step:
    python3 step5_predict.py generated/validation_set.csv generated/export.json
"""

import json
import numpy as np
import sys
from neural_network import NeuralNetMLP
from loss_functions import CategoricalCrossEntropy
from optimizers import Adam, SGD
from layers import Dense, ReLU, Softmax
from tools.utils import section, subsection


def build_layers_from_session(session):
    """Rebuild the exact same layer stack described in session.json."""
    layer = session["layer"]

    if layer is None:
        return [
            Dense(30, 24), ReLU(),
            Dense(24, 10), ReLU(),
            Dense(10,  8), ReLU(),
            Dense( 8,  2), Softmax()
        ]

    for n in layer:
        if n <= 0:
            raise ValueError("Every hidden layer must have at least 1 neuron.")

    layers_final = []
    last = layer[0]
    layers_final += [Dense(30, last), ReLU()]

    if len(layer) == 1:
        layers_final += [Dense(last, 2), Softmax()]
        return layers_final

    for i in range(1, len(layer)):
        if i == len(layer) - 1:
            layers_final += [Dense(last, 2), Softmax()]
        else:
            layers_final += [Dense(last, layer[i]), ReLU()]
            last = layer[i]

    return layers_final


def main():
    try:
        with open("generated/session.json") as f:
            session = json.load(f)
    except FileNotFoundError:
        print("  Error: session.json not found.")
        return

    required = ["generated/X_train.npy", "generated/y_train.npy",
                "generated/X_val.npy",   "generated/y_val.npy"]
    for fname in required:
        try:
            open(fname)
        except FileNotFoundError:
            print(f"  Error: {fname} not found.")
            return

    X_train = np.load("generated/X_train.npy")
    y_train = np.load("generated/y_train.npy")
    X_val   = np.load("generated/X_val.npy")
    y_val   = np.load("generated/y_val.npy")
    optimizer_name = "Adam" if session["adam"] else "SGD"

    try:
        model = NeuralNetMLP(build_layers_from_session(session))
    except ValueError as err:
        print(f"  Error: {err}")
        sys.exit(1)

    section("CONFIGURE TRAINING")
    print(f"""
  Attaching loss function and optimizer to the model.
""")

    if session["adam"]:
        model.configure_training(
            loss_criterion=CategoricalCrossEntropy(),
            weight_updater=Adam(build_layers_from_session(session),
                                lr=session["learning_rate"])
        )
    else:
        model.configure_training(
            loss_criterion=CategoricalCrossEntropy(),
            weight_updater=SGD(lr=session["learning_rate"])
        )

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
