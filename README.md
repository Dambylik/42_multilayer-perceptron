# Multilayer Perceptron — Breast Cancer Classification

A neural network built from scratch in Python, trained to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer dataset.

No neural network library is used — only NumPy for linear algebra and Matplotlib for visualization.

---

## Architecture

```
Input (30 features)
    │
    ▼
Dense(30 → 24) + ReLU
    │
    ▼
Dense(24 → 10) + ReLU
    │
    ▼
Dense(10 → 8)  + ReLU
    │
    ▼
Dense(8  → 2)  + Softmax   ← training  (CCE loss)
                 Sigmoid    ← inference (BCE loss, fused from Softmax weights)
```

The default architecture uses **3 hidden layers**. It is fully configurable via `--layer`.

At inference time, the 2-output Softmax head is converted to a single logit using the identity `Softmax([a,b])[0] = Sigmoid(a−b)`, so no Sigmoid layer is needed in the source.

---

## Files

```
├── data.csv                        # Raw dataset (569 samples, 32 columns)
├── main.py                         # Entry point: parse args, split dataset, save session.json
├── 1_load.py                       # Load & visualize dataset, save normalization stats
├── 2_configure.py                  # Display architecture and training config
├── 3_train.py                      # Build model, run training loop, save export.json
├── 4_predict.py                    # Load export.json, run inference, show confusion matrix
│
├── src/
│   ├── layers.py                   # Dense, ReLU, Softmax
│   ├── loss_functions.py           # BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── neural_network.py           # NeuralNetMLP: forward, backward, training loop, inference
│   └── optimizers.py               # SGD, Adam
│
├── tools/
│   ├── utils.py                    # Dataset loading, normalization, split, session helpers
│   ├── visualize_dataset.py        # Class distribution and correlation heatmap plots
│   ├── visualize_graphs.py         # Training curve plots
│   └── trace_one_sample.py         # Debug: trace a single sample through the network
│
└── generated/                      # Created at runtime
    ├── session.json                # Training configuration saved by main.py
    ├── train_set.csv               # Auto-split training set
    ├── validation_set.csv          # Auto-split validation set
    ├── export.json                 # Trained weights + topology
    └── norm_stats.json             # Mean/std used for normalization
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

The workflow is split into sequential steps. Run `main.py` first to configure the session, then run each step in order.

### Step 0 — Configure

```bash
python3 main.py data.csv
```

Parses arguments, splits `data.csv` into train/validation sets, and saves `generated/session.json`.

**Available options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | 0.2 | Validation ratio for auto-split |
| `--epochs` | 70 | Number of training epochs |
| `--batch_size` | 16 | Samples per gradient update |
| `--learning_rate` | 0.01 | Step size for weight updates |
| `--layer N [N ...]` | `24 10 8` | Hidden layer sizes |
| `--patience` | 3 | Early stopping patience |
| `--adam` | off | Use Adam optimizer instead of SGD |

---

### Step 1 — Load & Visualize

```bash
python3 1_load.py
```

Loads train/validation CSVs, applies z-score normalization, saves `norm_stats.json`, and plots dataset visualizations.

**Class distribution** — how many M vs B samples:

![Class distribution](images/class_distribution.png)

**Correlation heatmap** — which features are correlated across the training set:

![Training heatmap](images/training_heatmap.png)

---

### Step 2 — Review Configuration

```bash
python3 2_configure.py
```

Displays the layer architecture and training hyperparameters that will be used. No computation — display only.

---

### Step 3 — Train

```bash
python3 3_train.py
```

Builds the network, runs mini-batch training with backpropagation, and saves weights to `generated/export.json`.

**Training curves** — loss and accuracy over epochs:

![Training curves](images/training_curves.png)

---

### Step 4 — Predict

```bash
python3 4_predict.py generated/validation_set.csv generated/export.json
```

Loads `export.json`, fuses the Softmax head into a single sigmoid logit, runs inference on the validation set, and displays a confusion matrix.

**Confusion matrix:**

![Confusion matrix](images/confusion_matrix.png)

---

## How It Works

### Forward Propagation

Input data flows through each layer:

```
output = activation(input @ W + b)
```

Each Dense layer stores its input for the backward pass.

### Backpropagation

Gradients are computed using the chain rule, flowing from the loss backward through each layer:

- **Output layer (fused gradient):** for Softmax+CCE, the delta simplifies to `δ = (ŷ − y) / batch_size`
- **Hidden layers:** `δ_prev = δ @ W.T`, then `dW = input.T @ δ`, `db = sum(δ, axis=0)`

### Weight Initialization

Dense layers use **He uniform** initialization:

```
limit = sqrt(6 / n_in)
W ~ Uniform(−limit, +limit)
```

### Weight Update

SGD:
```
W = W − lr * dW
b = b − lr * db
```

With **Adam**, gradients are scaled by their running mean and variance (first and second moments) with bias correction.

### Early Stopping

Training stops when validation loss does not improve for `--patience` consecutive epochs, preventing overfitting.

---

## Results

| Metric | Value |
|--------|-------|
| Val Loss | ~0.17 |
| Val Accuracy | ~94.8% |

The off-diagonal cells of the confusion matrix show misclassifications:
- **FP** (predicted M, actually B) — false alarm
- **FN** (predicted B, actually M) — missed cancer, the more critical error to minimize
