"""
FULL MANUAL TRACE — one sample, 3 features, tiny network
=========================================================
Sample : row 1 of data.csv
         ID=842302, Label=M, features=[17.99, 10.38, 122.8]
Network: Input(3) → Dense(3→2)+ReLU → Dense(2→2)+Softmax → CCE Loss
Optimizer: SGD, lr=0.1
Seed: 42 (fully reproducible)
"""

import numpy as np

np.random.seed(42)
SEP = "─" * 70

def section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def subsection(title):
    print(f"\n  {'─'*60}")
    print(f"  ▶  {title}")
    print(f"  {'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 0 — RAW DATA (from data.csv, row 1)")
# ─────────────────────────────────────────────────────────────────────────────

raw_x1 = 17.99   # radius_mean
raw_x2 = 10.38   # texture_mean
raw_x3 = 122.80  # perimeter_mean
label  = "M"

print(f"""
  Row 1 of data.csv:
    ID      = 842302
    Label   = {label}  →  Malignant
    x₁ (radius_mean)    = {raw_x1}
    x₂ (texture_mean)   = {raw_x2}
    x₃ (perimeter_mean) = {raw_x3}

  We only use 3 of the 30 features to keep calculations readable.
  In the real project, all 30 are used identically.
""")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1 — STANDARDIZATION  (utils.py → standardize())")
# ─────────────────────────────────────────────────────────────────────────────

print("""
  Formula:   X_norm = (X - mean) / std
  means and stds are computed from the TRAINING set only.

  For this demo we use pre-computed training statistics
  (realistic values from the full Wisconsin dataset):
""")

means = np.array([14.13, 19.29, 91.97])
stds  = np.array([ 3.52,  4.30, 24.30])

print(f"  means = {means}")
print(f"  stds  = {stds}")
print()

x = np.array([[raw_x1, raw_x2, raw_x3]], dtype=float)  # shape (1,3)

x_norm = (x - means) / stds

print(f"  x₁_norm = ({raw_x1} - {means[0]}) / {stds[0]}")
print(f"          = {raw_x1 - means[0]:.4f} / {stds[0]}")
print(f"          = {x_norm[0,0]:.6f}")
print()
print(f"  x₂_norm = ({raw_x2} - {means[1]}) / {stds[1]}")
print(f"          = {raw_x2 - means[1]:.4f} / {stds[1]}")
print(f"          = {x_norm[0,1]:.6f}")
print()
print(f"  x₃_norm = ({raw_x3} - {means[2]}) / {stds[2]}")
print(f"          = {raw_x3 - means[2]:.4f} / {stds[2]}")
print(f"          = {x_norm[0,2]:.6f}")
print()
print(f"  → X_norm = {x_norm}")
print(f"            shape: {x_norm.shape}   (1 sample × 3 features)")

X = x_norm  # final normalized input

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 — ONE-HOT LABEL ENCODING  (utils.py → create_training_set())")
# ─────────────────────────────────────────────────────────────────────────────

print("""
  The network has 2 output neurons:  [neuron_M, neuron_B]
  We encode the label as a probability distribution:

    M  →  [1, 0]   (100% malignant, 0% benign)
    B  →  [0, 1]   (0% malignant, 100% benign)

  Label = "M"  →  y_true = [1, 0]
""")

y_true = np.array([[1, 0]], dtype=float)
print(f"  y_true = {y_true}   shape: {y_true.shape}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3 — WEIGHT INITIALIZATION  (layers.py → Dense.__init__())")
# ─────────────────────────────────────────────────────────────────────────────

print("""
  Formula:  limit = sqrt(6 / input_size)
            W ~ Uniform(-limit, +limit)
            b = zeros

  This is called "He uniform" initialization.
  It keeps activations in a reasonable range at the start.
""")

# --- Layer 1: Dense(3→2) ---
np.random.seed(42)
limit1 = np.sqrt(6 / 3)
W1 = np.random.uniform(-limit1, limit1, size=(3, 2))
b1 = np.zeros((1, 2))

print(f"  LAYER 1 — Dense(3 inputs → 2 neurons)")
print(f"    limit = sqrt(6 / 3) = sqrt(2) = {limit1:.6f}")
print(f"    W1 shape: (3 inputs, 2 neurons)")
print(f"    W1 =")
for i, row in enumerate(W1):
    print(f"         row {i}: [{row[0]:+.6f},  {row[1]:+.6f}]")
print(f"    b1 = {b1}  (zeros)")

# --- Layer 2: Dense(2→2) ---
limit2 = np.sqrt(6 / 2)
W2 = np.random.uniform(-limit2, limit2, size=(2, 2))
b2 = np.zeros((1, 2))

print()
print(f"  LAYER 2 — Dense(2 inputs → 2 neurons)")
print(f"    limit = sqrt(6 / 2) = {limit2:.6f}")
print(f"    W2 shape: (2 inputs, 2 neurons)")
print(f"    W2 =")
for i, row in enumerate(W2):
    print(f"         row {i}: [{row[0]:+.6f},  {row[1]:+.6f}]")
print(f"    b2 = {b2}  (zeros)")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4 — FORWARD PASS: LAYER 1  (layers.py → Dense.forward() + ReLU.forward())")
# ─────────────────────────────────────────────────────────────────────────────

subsection("Dense(3→2): Z1 = X · W1 + b1")

print(f"""
  X  (1×3):
    {X}

  W1 (3×2):
    col0       col1
    {W1[0,0]:+.6f}   {W1[0,1]:+.6f}    ← weight from x₁
    {W1[1,0]:+.6f}   {W1[1,1]:+.6f}    ← weight from x₂
    {W1[2,0]:+.6f}   {W1[2,1]:+.6f}    ← weight from x₃

  Matrix multiplication  X (1×3) · W1 (3×2) = Z1 (1×2):
""")

# Manual calculation neuron by neuron
x0, x1, x2 = X[0,0], X[0,1], X[0,2]

z1_n0 = x0*W1[0,0] + x1*W1[1,0] + x2*W1[2,0] + b1[0,0]
z1_n1 = x0*W1[0,1] + x1*W1[1,1] + x2*W1[2,1] + b1[0,1]

print(f"  Neuron 0:  z1₀ = x₁·W1[0,0] + x₂·W1[1,0] + x₃·W1[2,0] + b1[0]")
print(f"           = ({x0:.6f})·({W1[0,0]:+.6f})")
print(f"           + ({x1:.6f})·({W1[1,0]:+.6f})")
print(f"           + ({x2:.6f})·({W1[2,0]:+.6f})")
print(f"           + {b1[0,0]}")
print(f"           = {x0*W1[0,0]:.6f} + {x1*W1[1,0]:.6f} + {x2*W1[2,0]:.6f} + {b1[0,0]:.6f}")
print(f"           = {z1_n0:.6f}")
print()
print(f"  Neuron 1:  z1₁ = x₁·W1[0,1] + x₂·W1[1,1] + x₃·W1[2,1] + b1[1]")
print(f"           = ({x0:.6f})·({W1[0,1]:+.6f})")
print(f"           + ({x1:.6f})·({W1[1,1]:+.6f})")
print(f"           + ({x2:.6f})·({W1[2,1]:+.6f})")
print(f"           + {b1[0,1]}")
print(f"           = {x0*W1[0,1]:.6f} + {x1*W1[1,1]:.6f} + {x2*W1[2,1]:.6f} + {b1[0,1]:.6f}")
print(f"           = {z1_n1:.6f}")

Z1 = np.dot(X, W1) + b1
print(f"\n  Z1 = {Z1}  (numpy confirms)")

subsection("ReLU: A1 = max(0, Z1)")

a1_n0 = max(0.0, z1_n0)
a1_n1 = max(0.0, z1_n1)

print(f"""
  ReLU simply zeroes out any negative value:
    A1[0] = max(0, {z1_n0:.6f}) = {a1_n0:.6f}  {'← positive, passes through' if z1_n0>0 else '← negative, KILLED (set to 0)'}
    A1[1] = max(0, {z1_n1:.6f}) = {a1_n1:.6f}  {'← positive, passes through' if z1_n1>0 else '← negative, KILLED (set to 0)'}
""")

A1 = np.maximum(0, Z1)
mask1 = (Z1 > 0)   # stored for backward pass
print(f"  A1 = {A1}  (saved as activation output)")
print(f"  mask (Z1>0) = {mask1}  ← remembered for backprop!")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5 — FORWARD PASS: LAYER 2  (Dense.forward() + Softmax.forward())")
# ─────────────────────────────────────────────────────────────────────────────

subsection("Dense(2→2): Z2 = A1 · W2 + b2")

a1_0, a1_1 = A1[0,0], A1[0,1]

z2_n0 = a1_0*W2[0,0] + a1_1*W2[1,0] + b2[0,0]
z2_n1 = a1_0*W2[0,1] + a1_1*W2[1,1] + b2[0,1]

print(f"""
  A1 (1×2) = {A1}

  W2 (2×2):
    col0       col1
    {W2[0,0]:+.6f}   {W2[0,1]:+.6f}    ← weight from A1[0]
    {W2[1,0]:+.6f}   {W2[1,1]:+.6f}    ← weight from A1[1]

  Neuron 0:  z2₀ = a1₀·W2[0,0] + a1₁·W2[1,0] + b2[0]
           = ({a1_0:.6f})·({W2[0,0]:+.6f})
           + ({a1_1:.6f})·({W2[1,0]:+.6f})
           + {b2[0,0]}
           = {a1_0*W2[0,0]:.6f} + {a1_1*W2[1,0]:.6f}
           = {z2_n0:.6f}

  Neuron 1:  z2₁ = a1₀·W2[0,1] + a1₁·W2[1,1] + b2[1]
           = ({a1_0:.6f})·({W2[0,1]:+.6f})
           + ({a1_1:.6f})·({W2[1,1]:+.6f})
           + {b2[0,1]}
           = {a1_0*W2[0,1]:.6f} + {a1_1*W2[1,1]:.6f}
           = {z2_n1:.6f}
""")

Z2 = np.dot(A1, W2) + b2
print(f"  Z2 = {Z2}  (numpy confirms)")

subsection("Softmax: ŷ = exp(Z2) / sum(exp(Z2))")

# Numerical stability: subtract max
z2_max = np.max(Z2)
z2_stable = Z2 - z2_max

print(f"""
  Step 1 — Subtract max for numerical stability (avoids exp overflow):
    max(Z2)    = {z2_max:.6f}
    Z2_stable  = Z2 - {z2_max:.6f} = {z2_stable}

  Step 2 — Compute exp:
    exp(z2_stable[0]) = exp({z2_stable[0,0]:.6f}) = {np.exp(z2_stable[0,0]):.6f}
    exp(z2_stable[1]) = exp({z2_stable[0,1]:.6f}) = {np.exp(z2_stable[0,1]):.6f}
    sum of exp         = {np.sum(np.exp(z2_stable)):.6f}

  Step 3 — Divide by sum:
    ŷ[0] = {np.exp(z2_stable[0,0]):.6f} / {np.sum(np.exp(z2_stable)):.6f} = {np.exp(z2_stable[0,0])/np.sum(np.exp(z2_stable)):.6f}  ← P(Malignant)
    ŷ[1] = {np.exp(z2_stable[0,1]):.6f} / {np.sum(np.exp(z2_stable)):.6f} = {np.exp(z2_stable[0,1])/np.sum(np.exp(z2_stable)):.6f}  ← P(Benign)

  Check: {np.exp(z2_stable[0,0])/np.sum(np.exp(z2_stable)):.6f} + {np.exp(z2_stable[0,1])/np.sum(np.exp(z2_stable)):.6f} = {np.exp(z2_stable[0,0])/np.sum(np.exp(z2_stable)) + np.exp(z2_stable[0,1])/np.sum(np.exp(z2_stable)):.6f}  ✓ (always sums to 1)
""")

exp_z = np.exp(z2_stable)
y_pred = exp_z / np.sum(exp_z, axis=1, keepdims=True)
print(f"  ŷ = {y_pred}")
print(f"       shape: {y_pred.shape}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6 — LOSS  (loss_functions.py → CategoricalCrossEntropy.forward())")
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
  Formula:  L = -mean( Σ y_true * log(ŷ) )

  y_true = {y_true[0]}   (M=1, B=0)
  ŷ      = [{y_pred[0,0]:.6f}, {y_pred[0,1]:.6f}]

  For this single sample:
    term_M = y_true[M] * log(ŷ[M]) = {y_true[0,0]:.1f} * log({y_pred[0,0]:.6f})
           = 1 * {np.log(y_pred[0,0]):.6f}
           = {np.log(y_pred[0,0]):.6f}

    term_B = y_true[B] * log(ŷ[B]) = {y_true[0,1]:.1f} * log({y_pred[0,1]:.6f})
           = 0 * {np.log(y_pred[0,1]):.6f}
           = 0.0   ← zero because the true label is M, not B

    sum = {np.log(y_pred[0,0]):.6f} + 0.0 = {np.log(y_pred[0,0]):.6f}

    L = -mean({np.log(y_pred[0,0]):.6f}) = {-np.mean(np.log(y_pred[0,0])):.6f}
""")

y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
loss = -np.mean(np.sum(y_true * np.log(y_pred_clip), axis=1))
print(f"  Loss L = {loss:.6f}")

if loss > 0.5:
    print(f"""
  Interpretation:
    The network predicts P(M)={y_pred[0,0]:.4f} but truth is 1.0
    → It is {'confident but wrong' if y_pred[0,1] > 0.6 else 'uncertain'}
    → Loss {loss:.4f} is {'high — network needs significant correction' if loss > 1 else 'moderate'}
""")
else:
    print(f"""
  Interpretation:
    The network predicts P(M)={y_pred[0,0]:.4f} and truth is 1.0
    → It's doing well already!
    → Loss {loss:.4f} is low
""")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7 — BACKWARD PASS  (the gradients)")
# ─────────────────────────────────────────────────────────────────────────────

print("""
  Goal: find how much each weight contributed to the error.
  We propagate the error BACKWARDS through each layer using the chain rule.
""")

subsection("Fused gradient: δ = (ŷ - y_true) / batch_size  [Softmax + CCE]")

batch_size = 1
delta = (y_pred_clip - y_true) / batch_size

print(f"""
  This is the "fused gradient" shortcut (neural_network.py:23-25).
  When Softmax is paired with CategoricalCrossEntropy, the full Jacobian
  simplifies to:

    δ = (ŷ - y_true) / N

  where N = batch size = {batch_size}

    δ[M] = ŷ[M] - y_true[M] = {y_pred_clip[0,0]:.6f} - {y_true[0,0]:.1f} = {delta[0,0]:.6f}
    δ[B] = ŷ[B] - y_true[B] = {y_pred_clip[0,1]:.6f} - {y_true[0,1]:.1f} = {delta[0,1]:.6f}

  δ = {delta}

  Intuition:
    δ[M] = {delta[0,0]:.6f}  →  {'negative → we need to INCREASE P(M)' if delta[0,0] < 0 else 'positive → we need to DECREASE P(M)'}
    δ[B] = {delta[0,1]:.6f}  →  {'positive → we need to DECREASE P(B)' if delta[0,1] > 0 else 'negative → we need to INCREASE P(B)'}
""")

subsection("Dense Layer 2 backward  (layers.py → Dense.backward())")

print(f"""
  Received gradient (from softmax fused):  δ = {delta}
  Stored input was: A1 = {A1}

  Gradient w.r.t. W2:
    dW2 = A1ᵀ · δ      (shape: (2,1) · (1,2) = (2,2))

    A1ᵀ = {A1.T}

    dW2[0,0] = A1[0] * δ[0] = {A1[0,0]:.6f} * {delta[0,0]:.6f} = {A1[0,0]*delta[0,0]:.6f}
    dW2[0,1] = A1[0] * δ[1] = {A1[0,0]:.6f} * {delta[0,1]:.6f} = {A1[0,0]*delta[0,1]:.6f}
    dW2[1,0] = A1[1] * δ[0] = {A1[0,1]:.6f} * {delta[0,0]:.6f} = {A1[0,1]*delta[0,0]:.6f}
    dW2[1,1] = A1[1] * δ[1] = {A1[0,1]:.6f} * {delta[0,1]:.6f} = {A1[0,1]*delta[0,1]:.6f}
""")

dW2 = np.dot(A1.T, delta)
print(f"  dW2 = {dW2}")

db2 = np.sum(delta, axis=0, keepdims=True)
print(f"""
  Gradient w.r.t. b2:
    db2 = Σδ (sum over batch) = {delta} = {db2}
""")

delta_back_L2 = np.dot(delta, W2.T)
print(f"""
  Gradient to pass BACK to previous layer:
    δ_prev = δ · W2ᵀ      (shape: (1,2) · (2,2) = (1,2))

    W2ᵀ =
      {W2.T[0]}
      {W2.T[1]}

    δ_prev[0] = δ[0]*W2[0,0] + δ[1]*W2[0,1]
              = {delta[0,0]:.6f}*{W2[0,0]:.6f} + {delta[0,1]:.6f}*{W2[0,1]:.6f}
              = {delta[0,0]*W2[0,0]:.6f} + {delta[0,1]*W2[0,1]:.6f}
              = {delta_back_L2[0,0]:.6f}

    δ_prev[1] = δ[0]*W2[1,0] + δ[1]*W2[1,1]
              = {delta[0,0]:.6f}*{W2[1,0]:.6f} + {delta[0,1]:.6f}*{W2[1,1]:.6f}
              = {delta[0,0]*W2[1,0]:.6f} + {delta[0,1]*W2[1,1]:.6f}
              = {delta_back_L2[0,1]:.6f}

  δ_prev = {delta_back_L2}
""")

subsection("ReLU backward  (layers.py → ReLU.backward())")

print(f"""
  ReLU backward: only pass gradient where input was > 0 (the saved mask).

  mask (Z1 > 0) = {mask1}
  δ from Dense2 = {delta_back_L2}

  δ_relu = δ_prev * mask
         = [{delta_back_L2[0,0]:.6f} * {int(mask1[0,0])},  {delta_back_L2[0,1]:.6f} * {int(mask1[0,1])}]
         = [{delta_back_L2[0,0]*int(mask1[0,0]):.6f}, {delta_back_L2[0,1]*int(mask1[0,1]):.6f}]
""")

delta_relu = delta_back_L2 * mask1
print(f"  δ_relu = {delta_relu}")

if not mask1[0,0] or not mask1[0,1]:
    print(f"""
  NOTE: One or more gradients were KILLED by ReLU!
  This happens when Z1 ≤ 0 — ReLU outputs 0, so there's no gradient to pass back.
  (This is the "dead neuron" phenomenon)
""")

subsection("Dense Layer 1 backward  (layers.py → Dense.backward())")

print(f"""
  Received gradient: δ_relu = {delta_relu}
  Stored input was:  X = {X}  (the normalized sample)

  Gradient w.r.t. W1:
    dW1 = Xᵀ · δ_relu      (shape: (3,1) · (1,2) = (3,2))

    Xᵀ = {X.T}

""")

dW1 = np.dot(X.T, delta_relu)
for i in range(3):
    for j in range(2):
        print(f"    dW1[{i},{j}] = X[{i}] * δ[{j}] = {X[0,i]:.6f} * {delta_relu[0,j]:.6f} = {X[0,i]*delta_relu[0,j]:.6f}")

print(f"\n  dW1 =")
for row in dW1:
    print(f"    {row}")

db1 = np.sum(delta_relu, axis=0, keepdims=True)
print(f"""
  Gradient w.r.t. b1:
    db1 = Σδ_relu = {db1}
""")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8 — WEIGHT UPDATE  (optimizers.py → SGD.step())")
# ─────────────────────────────────────────────────────────────────────────────

lr = 0.1

print(f"""
  Formula:  W_new = W_old - learning_rate * dW
  Learning rate (lr) = {lr}
""")

subsection("Update W1")

W1_new = W1 - lr * dW1
print(f"  {'Weight':<10}  {'Old value':>12}  {'Gradient':>12}  {'lr*grad':>12}  {'New value':>12}")
print(f"  {'─'*62}")
for i in range(3):
    for j in range(2):
        print(f"  W1[{i},{j}]    {W1[i,j]:>12.6f}  {dW1[i,j]:>12.6f}  {lr*dW1[i,j]:>12.6f}  {W1_new[i,j]:>12.6f}")

b1_new = b1 - lr * db1
print(f"\n  {'Bias':<10}  {'Old value':>12}  {'Gradient':>12}  {'lr*grad':>12}  {'New value':>12}")
print(f"  {'─'*62}")
for j in range(2):
    print(f"  b1[{j}]      {b1[0,j]:>12.6f}  {db1[0,j]:>12.6f}  {lr*db1[0,j]:>12.6f}  {b1_new[0,j]:>12.6f}")

subsection("Update W2")

W2_new = W2 - lr * dW2
print(f"  {'Weight':<10}  {'Old value':>12}  {'Gradient':>12}  {'lr*grad':>12}  {'New value':>12}")
print(f"  {'─'*62}")
for i in range(2):
    for j in range(2):
        print(f"  W2[{i},{j}]    {W2[i,j]:>12.6f}  {dW2[i,j]:>12.6f}  {lr*dW2[i,j]:>12.6f}  {W2_new[i,j]:>12.6f}")

b2_new = b2 - lr * db2
print(f"\n  {'Bias':<10}  {'Old value':>12}  {'Gradient':>12}  {'lr*grad':>12}  {'New value':>12}")
print(f"  {'─'*62}")
for j in range(2):
    print(f"  b2[{j}]      {b2[0,j]:>12.6f}  {db2[0,j]:>12.6f}  {lr*db2[0,j]:>12.6f}  {b2_new[0,j]:>12.6f}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9 — VERIFY: FORWARD PASS WITH UPDATED WEIGHTS")
# ─────────────────────────────────────────────────────────────────────────────

Z1_new = np.dot(X, W1_new) + b1_new
A1_new = np.maximum(0, Z1_new)
Z2_new = np.dot(A1_new, W2_new) + b2_new
exp_z_new = np.exp(Z2_new - np.max(Z2_new))
y_pred_new = exp_z_new / np.sum(exp_z_new, axis=1, keepdims=True)
y_pred_new_clip = np.clip(y_pred_new, 1e-7, 1 - 1e-7)
loss_new = -np.mean(np.sum(y_true * np.log(y_pred_new_clip), axis=1))

print(f"""
  After one gradient step, running the same sample again:

  Old prediction:  ŷ = [{y_pred[0,0]:.6f},  {y_pred[0,1]:.6f}]
  New prediction:  ŷ = [{y_pred_new[0,0]:.6f},  {y_pred_new[0,1]:.6f}]
  True label:      y = [1.0,  0.0]   (Malignant)

  Old loss: {loss:.6f}
  New loss: {loss_new:.6f}
  Change:   {loss_new - loss:+.6f}  {'↓ IMPROVED ✓' if loss_new < loss else '↑ got worse (normal for 1 sample)'}

  P(Malignant) went from {y_pred[0,0]:.4f} → {y_pred_new[0,0]:.4f}
  {'→ The network is moving in the RIGHT direction!' if y_pred_new[0,0] > y_pred[0,0] else '→ Small step, will improve over many epochs.'}

  This was ONE sample, ONE forward+backward pass.
  In real training, this repeats thousands of times across all samples.
""")

# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY — Full data flow for one sample")
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  RAW INPUT                                                      │
  │  [17.99, 10.38, 122.8]  label=M                                 │
  │                      ↓  standardize()                           │
  │  NORMALIZED: {X[0]}              │
  │                      ↓  Dense(3→2) + ReLU                       │
  │  A1: {A1[0]}                               │
  │                      ↓  Dense(2→2) + Softmax                    │
  │  ŷ:  [{y_pred[0,0]:.4f}, {y_pred[0,1]:.4f}]   P(M), P(B)           │
  │                      ↓  CCE loss                                │
  │  Loss = {loss:.6f}                                        │
  │                      ↓  backward (chain rule)                   │
  │  δ_softmax = {delta[0]}                   │
  │  dW2 = {dW2[0]}  ...          │
  │  dW1 = ...                                                      │
  │                      ↓  SGD update  W ← W - 0.1·dW             │
  │  NEW Loss = {loss_new:.6f}  ({'↓ better' if loss_new < loss else '≈ same'})                              │
  └─────────────────────────────────────────────────────────────────┘
""")

print("Done! All calculations are exact — verified by numpy at each step.")
