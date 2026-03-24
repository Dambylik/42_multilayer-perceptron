"""
STEP 5 - PREDICT
"""

import json
import numpy as np
import argparse
from layers import Dense, ReLU, Sigmoid, Softmax
from neural_network import NeuralNetMLP
from loss_functions import BinaryCrossEntropy
from optimizers import SGD
from tools.utils import create_set_sigmoid, section, subsection


def reconstruct_model_from_json(serialized_data):
    """
    Rebuild the network from export.json produced by main.py.

    For Dense layers:  restore exact weight/bias matrices from the saved lists.
    For activations:   instantiate a fresh ReLU / Sigmoid / Softmax object.
    """
    layers_list    = []
    activation_map = {"ReLU": ReLU, "Sigmoid": Sigmoid, "Softmax": Softmax}

    print(f"\n    Reconstructing layers from JSON:")
    for idx, layer_config in enumerate(serialized_data):
        layer_type = layer_config["type"]

        if layer_type == "Dense":
            in_dim  = len(layer_config["W"])
            out_dim = len(layer_config["W"][0])
            new_layer         = Dense(in_dim, out_dim)
            new_layer.weights = np.array(layer_config["W"])
            new_layer.biases  = np.array(layer_config["b"])
            layers_list.append(new_layer)
            print(f"      Layer {idx}: Dense ({in_dim}→{out_dim})  ← weights RESTORED from JSON")
        elif layer_type in activation_map:
            layers_list.append(activation_map[layer_type]())
            print(f"      Layer {idx}: {layer_type}")

    return layers_list


# ──────────────────────────────────────────────────────────────────────────────

def run_inference():

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 0 — ARGUMENTS
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 0 — ARGUMENTS")

    cli_parser = argparse.ArgumentParser(description="MLP Prediction Utility")
    cli_parser.add_argument("data_file",  nargs="?", default="generated/validation_set.csv")
    cli_parser.add_argument("model_file", nargs="?", default="generated/export.json")
    args = cli_parser.parse_args()

    print(f"""
  Data file  : {args.data_file}
  Model file : {args.model_file}
""")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — LOAD MODEL FROM export.json
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 1 — LOAD MODEL  (predict.py → reconstruct_model_from_json())")
    print(f"""
  export.json stores each layer as a JSON object:
    Dense layers   → carry "W" (weights matrix) and "b" (biases vector)
    Activation layers → just the type name ("ReLU", "Softmax", ...)

  We rebuild the exact same layer stack in memory and inject the saved
  weight matrices — so the network is in the same state as at the end
  of training.
""")

    try:
        with open(args.model_file, "r") as json_file:
            reconstructed_layers = reconstruct_model_from_json(json.load(json_file))
    except Exception as error:
        print(f"  Error loading model: {error}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — LOAD & PREPROCESS VALIDATION DATA
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 2 — LOAD VALIDATION DATA  (utils.py → create_set_sigmoid())")
    print(f"""
  For inference we use Sigmoid (single output) instead of Softmax (2 outputs).
  Labels are encoded as:
    M (Malignant) → [[1]]   B (Benign) → [[0]]

  NOTE: we do NOT recompute means/stds from this file.
  In a real deployment the training statistics would be stored alongside
  the model and reused here.  For this project validation_set.csv was
  standardised with the training stats at split time, so the raw values
  are already on the correct scale.
""")

    try:
        val_features, val_labels, _, _ = create_set_sigmoid(args.data_file)
    except Exception as error:
        print(f"  Error loading data: {error}")
        return

    print(f"\n    Validation set shape : {val_features.shape}")
    print(f"    Label vector  shape  : {val_labels.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — SOFTMAX → SIGMOID WEIGHT FUSION
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 3 — SOFTMAX → SIGMOID CONVERSION  (predict.py)")
    print(f"""
  The model was TRAINED with a 2-neuron Softmax output head:
    ŷ = Softmax([z_M, z_B])     z_M = X·W[:,0] + b[0]
                                z_B = X·W[:,1] + b[1]

  For INFERENCE we convert to a 1-neuron Sigmoid output head:
    ŷ = Sigmoid(z_fused)        z_fused = X·W_fused + b_fused

  Why can we do this?  Mathematical equivalence:
    Softmax([a, b])[0]  =  exp(a) / (exp(a) + exp(b))
                        =  1 / (1 + exp(b−a))
                        =  Sigmoid(a − b)

  So we simply FUSE the two output columns into one:
    W_fused = W[:,0] − W[:,1]     (shape: n_inputs → 1)
    b_fused = b[0]   − b[1]       (scalar)

  Result: same predictions, half the output neurons, cleaner inference.
""")

    last_dense = reconstructed_layers[-2]   # the Dense before the final activation

    W_old = last_dense.weights
    b_old = last_dense.biases

    print(f"  Last Dense layer before fusion:")
    print(f"    W shape : {W_old.shape}   (2 output columns)")
    print(f"    b shape : {b_old.shape}")
    print(f"\n  Computing fused weights:")
    print(f"    W_fused = W[:,0] − W[:,1]  →  shape: ({W_old.shape[0]}, 1)")
    print(f"    b_fused = b[0,0] − b[0,1]")
    print(f"            = {b_old[0,0]:.6f} − {b_old[0,1]:.6f} = {b_old[0,0]-b_old[0,1]:.6f}")

    fused_weights = (last_dense.weights[:, 0] - last_dense.weights[:, 1]).reshape(-1, 1)
    fused_bias    = last_dense.biases[0][0] - last_dense.biases[0][1]

    last_dense.weights = fused_weights
    last_dense.biases  = np.array([[fused_bias]])

    reconstructed_layers[-1] = Sigmoid()   # swap Softmax → Sigmoid

    print(f"\n  After fusion:")
    print(f"    W_fused shape : {fused_weights.shape}   ← 1 output column")
    print(f"    b_fused       : {fused_bias:.6f}")
    print(f"    Output layer  : Sigmoid  (replaces Softmax)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — CONFIGURE INFERENCE MODEL
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 4 — CONFIGURE INFERENCE MODEL")
    print(f"""
  We build a NeuralNetMLP with the fused layers and attach:
    • BinaryCrossEntropy  — loss for a single-output Sigmoid (inference)
    • SGD (lr=0.1)        — required by configure_training() interface,
                            but NO weight updates happen during inference
                            (fit_predict does not call backward or _update_weights)
""")

    inference_model = NeuralNetMLP(reconstructed_layers)
    inference_model.configure_training(
        loss_criterion=BinaryCrossEntropy(),
        weight_updater=SGD(lr=0.1)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — RUN INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 5 — RUN INFERENCE  (neural_network.py → fit_predict())")
    print(f"""
  For each sample:
    1. Forward pass through the fused network
       Input(30) → Dense+ReLU → ... → Dense(→1) → Sigmoid → ŷ ∈ (0,1)
    2. Decision threshold: ŷ ≥ 0.5  →  Malignant     ŷ < 0.5  →  Benign
    3. Compare prediction to true label → build confusion matrix

  No backpropagation, no weight updates — read-only pass.
""")

    inference_model.fit_predict(val_features, val_labels, batch_size=4, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_inference()
