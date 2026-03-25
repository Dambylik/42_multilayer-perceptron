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
    history = model.execute_training(
        X_train, y_train,
        X_val, y_val,
        epochs=session["epochs"],
        batch_size=session["batch_size"],
        early_stopping_patience=session["patience"]
    )
    histories_path = "generated/histories.json"
    try:
        with open(histories_path) as f:
            all_runs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_runs = []

    optimizer_name = "Adam" if session.get("adam") else "SGD"
    layer_str      = str(session.get("layer") or [24, 10, 8])
    run_label      = (f"Run {len(all_runs)+1} | lr={session['learning_rate']} "
                      f"opt={optimizer_name} layers={layer_str}")
    all_runs.append({"label": run_label, **history})

    with open(histories_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"    Run saved to {histories_path}  ({len(all_runs)} total runs)")

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

    # Save a numbered copy only if session config changed from the previous run
    prev_runs = all_runs[:-1]  # all runs except the one just appended
    if prev_runs:
        prev_label = prev_runs[-1]["label"]
        curr_label = all_runs[-1]["label"]
        config_changed = prev_label != curr_label
    else:
        config_changed = False  # first run — no need for a numbered copy

    if config_changed:
        run_number    = len(all_runs)
        numbered_path = f"generated/export_run{run_number}.json"
        with open(numbered_path, "w") as f:
            json.dump(export, f, indent=4)
        print(f"    Config changed — also saved as {numbered_path}")


if __name__ == "__main__":
    main()
