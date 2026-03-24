"""
STEP 3 — CONFIGURE TRAINING

This step is DISPLAY ONLY — it explains the training configuration:
  • Which loss function will be used and why
  • Which optimizer will be used and why
  • What the fused gradient shortcut is
"""

import json
import numpy as np
from tools.utils import section, subsection


def main():
    try:
        with open("generated/session.json") as f:
            session = json.load(f)
    except FileNotFoundError:
        print("  Error: session.json not found.")
        return

    optimizer_name = "Adam" if session["adam"] else "SGD"
    lr             = session["learning_rate"]
    patience       = session["patience"]
    epochs         = session["epochs"]
    batch_size     = session["batch_size"]

    section("CONFIGURE TRAINING")
    subsection("Loss function : CategoricalCrossEntropy")
    print(f"""
  Formula:
    L = -mean( Σ_c  y_c · log(ŷ_c) )

  Where:
    y_c   → true probability for class c  (1 if true class, 0 otherwise)
    ŷ_c   → predicted probability for class c  (output of Softmax)
    Σ_c   → sum over all classes (M and B)
    log   → natural logarithm
    mean  → average over the batch

  Because y_c is 0 for all classes except the true one, the sum collapses:
    For a Malignant sample:  L = -log( ŷ[M] )
    For a Benign sample:     L = -log( ŷ[B] )

  Why -log?
    -log(1.0) = 0.0   → perfect prediction, zero loss
    -log(0.5) = 0.693 → uncertain, moderate loss
    -log(0.1) = 2.303 → confident but wrong, heavy penalty
    -log(0.01)= 4.605 → very wrong, enormous penalty
""")

    subsection(f"Optimizer — {optimizer_name}  (lr = {lr})")
    if not session["adam"]:
        print(f"""
  SGD — Stochastic Gradient Descent

  Update rule:
    W  ←  W  -  lr · dW
    b  ←  b  -  lr · db

  How to read it:
    dW  → gradient of the loss (tells us the direction of steepest ascent)
    lr  → learning rate = {lr}  (how big a step we take)
    The minus sign: we move OPPOSITE to the gradient → descend toward lower loss
""")
    else:
        print(f"""
  Adam — Adaptive Moment Estimation

  Adam keeps two running averages per parameter:

    m  ← β₁ · m + (1-β₁) · dW          ← 1st moment: direction (mean of gradients)
    v  ← β₂ · v + (1-β₂) · dW²         ← 2nd moment: magnitude (variance of gradients)

  Bias-corrected estimates (important in early steps when m and v ≈ 0):
    m̂  = m / (1 - β₁ᵗ)
    v̂  = v / (1 - β₂ᵗ)

  Update rule:
    W  ←  W  -  lr · m̂ / (√v̂ + ε)

  Hyperparameters:
    lr  = {lr}
    β₁  = 0.9     (how much we trust the old gradient direction)
    β₂  = 0.999   (how much we trust the old gradient magnitude)
    ε   = 1e-8    (prevents division by zero)

  Why is Adam better than SGD?
    → Each weight gets its OWN effective learning rate.
    → Weights with consistently large gradients get a smaller effective lr
       (preventing wild oscillations).
    → Weights with small gradients get a larger effective lr
       (speeding up progress on flat surfaces).
    → Faster convergence, more robust to lr choice.
""")

    subsection("Fused gradient — Softmax + CategoricalCrossEntropy")
    print(f"""
  Normally backpropagation through Softmax requires computing a
  full n×n Jacobian matrix (for n output classes):

    ∂ŷ_i/∂z_j = ŷ_i(δ_ij − ŷ_j)   ← expensive, O(n²) work

  When Softmax is paired with CategoricalCrossEntropy, this entire
  Jacobian multiplied by the CCE gradient collapses to:

    δ = (ŷ − y_true) / batch_size

  How?  Chain rule applied symbolically:
    ∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z
           = ( −y/ŷ ) · ( ŷ(δ−ŷ) )   [lots of terms cancel]
           = ŷ − y                    [after simplification]

  Result:
    • Same mathematical result as the full Jacobian
    • O(n) work instead of O(n²)
    • Much simpler code
    • The code just passes δ = (ŷ−y)/N straight through the Softmax layer

  This is why neural_network.py checks isinstance(Softmax) + isinstance(CCE)
  and sets use_fused_gradient = True on the output layer.
""")

    subsection("Training loop preview")
    n_train = None
    try:
        import os
        if os.path.exists("generated/X_train.npy"):
            n_train = np.load("generated/X_train.npy").shape[0]
    except Exception:
        pass

    n_batches_str = f"{(n_train + batch_size - 1) // batch_size}" if n_train else "?"

    print(f"""
  Configuration summary:
    Epochs         : {epochs}
    Batch size     : {batch_size}   →  ~{n_batches_str} batches per epoch
    Optimizer      : {optimizer_name}  (lr={lr})
    Early stopping : patience = {patience} epochs

  What happens each epoch:
    ┌──────────────────────────────────────────────────────────────┐
    │  for each mini-batch of {batch_size} samples:                │
    │    1. forward(X_batch)       → ŷ  (predictions)              │
    │    2. criterion.forward(ŷ,y) → L  (loss scalar)              │
    │    3. criterion.backward()   → δ  (fused gradient)           │
    │    4. backward(δ)            → dW, db  (layer gradients)     │
    │    5. _update_weights()      → W ← W - lr·dW                 │
    │                                                              │
    │  evaluate(X_val, y_val)     → val_loss, val_acc (read-only)  │
    │  early stopping check                                        │
    └──────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
