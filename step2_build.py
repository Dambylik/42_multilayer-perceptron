"""
STEP 2 — BUILD MODEL ARCHITECTURE

This step is DISPLAY ONLY — it builds the layer objects in memory,
prints the full architecture with shapes and parameter counts, then
discards them.  The real model is built (and trained) in step4_train.py.
"""

import json
import numpy as np
from layers import Dense, ReLU, Softmax
from tools.utils import section, subsection


def build_layers_from_session(session):
    """Construct the layer list from session.json."""
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

    arch_str = str(session["layer"]) if session["layer"] else "[24, 10, 8]  (default)"

    section("BUILD MODEL ARCHITECTURE : Dense / ReLU / Softmax)")
    print(f"""
  Requested hidden layers : {arch_str}

  Architecture rules:
    • Input is always 30 neurons  (one per normalised feature)
    • Hidden layers always use ReLU activation
    • Output is always 2 neurons  (one score per class: M and B)
    • Output activation is always Softmax during training

1. Dense layers
      Each output neuron connects to EVERY input neuron.
      Formula: Z = X · W + b
        X  → input  (batch x n_in)
        W  → weight matrix  (n_in x n_out)  ← the LEARNABLE part
        b  → bias vector    (1 x n_out)     ← the LEARNABLE part
        Z  → pre-activation output (batch x n_out)

2. ReLU hidden activation
      Formula: a = max(0, Z)
      Why not Sigmoid everywhere?
        Sigmoid squashes values into (0,1), causing VANISHING GRADIENTS
        ReLU simply thresholds at 0: no squashing for positive values,
        so gradients flow freely through them.

3. 2-neuron Softmax output
      Formula: ŷ_c = exp(z_c) / Σ exp(z_k)
      Converts raw scores (logits) into a probability distribution.
      ŷ[0] = P(Malignant)   ŷ[1] = P(Benign)   and ŷ[0]+ŷ[1] = 1.

4. Weight initialisation: LeCun uniform
      W ~ Uniform(−√(6/n_in), +√(6/n_in))
      Keeps the variance of activations stable across layers at the start.
      Avoids both exploding and vanishing activations before any training.
""")

    try:
        layers = build_layers_from_session(session)
    except ValueError as err:
        print(f"  Error: {err}")
        return

    subsection("Architecture summary")

    total_params = 0
    print(f"\n  {'#':<4}  {'Type':<10}  {'Input → Output':^18}  {'W shape':^14}  {'Params':>8}  Note")
    print(f"  {'─'*72}")

    for i, layer in enumerate(layers):
        name = type(layer).__name__
        if name == "Dense":
            n_in, n_out = layer.weights.shape
            shape_str   = f"({n_in} → {n_out})"
            w_shape     = f"({n_in}×{n_out})"
            p           = layer.weights.size + layer.biases.size
            total_params += p
            note = "<- output layer" if i == len(layers) - 2 else ""
            print(f"  {i:<4}  {name:<10}  {shape_str:^18}  {w_shape:^14}  {p:>8}  {note}")
        elif name == "ReLU":
            print(f"  {i:<4}  {name:<10}  {'max(0, z)':^18}  {'—':^14}  {'—':>8}")
        elif name == "Softmax":
            print(f"  {i:<4}  {name:<10}  {'exp(z)/Σexp':^18}  {'—':^14}  {'—':>8}  <- training output")

    print(f"  {'─'*72}")
    print(f"  {'Total trainable parameters':>50} : {total_params}")


if __name__ == "__main__":
    main()
