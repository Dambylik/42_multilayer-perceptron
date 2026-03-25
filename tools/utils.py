import json
import numpy as np
import csv
import pandas as pd
from src.layers import Dense, ReLU, Softmax


def load_session():
    """Load generated/session.json and return the dict, or None on error."""
    try:
        with open("generated/session.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("  Error: session.json not found.")
        return None


def build_optimizer(session, layers):
    """Build and return Adam or SGD from session config."""
    from src.optimizers import Adam, SGD
    lr = session["learning_rate"]
    if session["adam"]:
        return Adam(layers, lr=lr)
    return SGD(lr=lr)


def reconstruct_model_from_json(path):
    """Rebuild the layer stack from export.json. Returns list of layers."""
    with open(path, "r") as f:
        serialized = json.load(f)

    activation_map = {"ReLU": ReLU, "Softmax": Softmax}
    layers = []
    for idx, cfg in enumerate(serialized):
        if cfg["type"] == "Dense":
            n_in, n_out   = len(cfg["W"]), len(cfg["W"][0])
            layer         = Dense(n_in, n_out)
            layer.weights = np.array(cfg["W"])
            layer.biases  = np.array(cfg["b"])
            layers.append(layer)
        elif cfg["type"] in activation_map:
            layers.append(activation_map[cfg["type"]]())
    return layers


def fuse_softmax_to_sigmoid(layers):
    """
    Fuse the 2-output Softmax head into a single logit for sigmoid inference.
    Uses the identity: Softmax([a,b])[0] = Sigmoid(a-b)
    Fuses W_fused = W[:,0]-W[:,1] and b_fused = b[0,0]-b[0,1], then removes
    the Softmax layer — sigmoid is applied inline in fit_predict.
    """
    last_dense         = layers[-2]
    W, b               = last_dense.weights, last_dense.biases
    last_dense.weights = (W[:, 0] - W[:, 1]).reshape(-1, 1)
    last_dense.biases  = np.array([[b[0, 0] - b[0, 1]]])
    layers.pop()   # remove Softmax — no activation layer needed
    return layers


def build_inference_model(layers):
    """Wrap layers in a NeuralNetMLP ready for inference (no optimizer needed)."""
    from src.neural_network import NeuralNetMLP
    return NeuralNetMLP(layers)


def build_layers_from_session(session):
    """Build the layer stack described in session.json."""
    layer = session["layer"]

    if layer is None:
        return [
            Dense(30, 24), ReLU(),
            Dense(24, 10), ReLU(),
            Dense(10,  8), ReLU(),
            Dense( 8,  2), Softmax(),
        ]

    for n in layer:
        if n <= 0:
            raise ValueError("Every hidden layer must have at least 1 neuron.")

    layers_final = [Dense(30, layer[0]), ReLU()]
    last = layer[0]

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


def section(title):
    print(f"\n{'═'*70}\n  {title}\n{'═'*70}")


def subsection(title):
    print(f"\n  {'─'*60}\n  ▶  {title}\n  {'─'*60}")


def standardize(X, means=None, stds=None, epsilon=1e-8):
    """
    Z-score standardization:  X_norm = (X - mean) / std
    """
    if X.size == 0:
        return X, means, stds
    if means is None:
        means = np.mean(X, axis=0)
    if stds is None:
        stds = np.std(X, axis=0)
    X_norm = (X - means) / (stds + epsilon)
    print(f"    Normed vals: min={X_norm.min():.4f}  max={X_norm.max():.4f}  mean={X_norm.mean():.4f}")
    return X_norm, means, stds


def shuffle_data(X, y):
    """Shuffle features and labels together while keeping pair alignment."""
    indices = np.random.permutation(X.shape[0])
    return X[indices], y[indices]


def load_raw_dataset(csv_path):
    """
    Read CSV and return:
      X      — float feature matrix  (n_samples, 30)
      labels — list of raw strings   ['M', 'B', ...]
    """
    print(f"\n    Loading file : {csv_path}")

    X_list = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            X_list.append(row[2:])     # cols 2-31 = features
            labels.append(row[1])      # col 1 = label

    X = np.array([[float(x) for x in row] for row in X_list], dtype=float)

    n_M = labels.count("M")
    n_B = labels.count("B")
    print(f"    Rows read    : {len(labels)}")
    print(f"    Features     : {X.shape[1]} ")
    print(f"    Malignant (M): {n_M}   Benign (B): {n_B}")
    print(f"    Feature range: min={X.min():.4f}  max={X.max():.4f}")

    return X, labels


def load_dataset(csv_path, means=None, stds=None, one_hot=True):
    """
    Load CSV, encode labels, and standardize features.

    one_hot=True  (training)  : M → [1,0]   B → [0,1]   shape (n, 2)
    one_hot=False (inference) : M → [1]     B → [0]     shape (n, 1)
    """
    X, labels = load_raw_dataset(csv_path)
    if one_hot:
        y = np.array([[1, 0] if l == "M" else [0, 1] for l in labels], dtype=float)
    else:
        y = np.array([[1] if l == "M" else [0] for l in labels], dtype=float)
    X_norm, means, stds = standardize(X, means, stds)
    return X_norm, y, means, stds


def split_dataset(csv_path, validation_ratio,
                  train_out="generated/train_set.csv", val_out="generated/validation_set.csv"):
    """
    Randomly split a CSV into train and validation files.
    """
    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")

    data = pd.read_csv(csv_path, header=None)
    total = len(data)

    print(f"    Source file      : {csv_path}")
    print(f"    Total samples    : {total}")
    print(f"    Split ratio      : {int((1-validation_ratio)*100)}% train  /  "
          f"{int(validation_ratio*100)}% validation")

    validation_set = data.sample(frac=validation_ratio)
    train_set      = data.drop(validation_set.index)
    validation_set.to_csv(val_out,   index=False, header=False)
    train_set.to_csv(train_out, index=False, header=False)
