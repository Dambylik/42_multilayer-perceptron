# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Visualize dataset (before training)
python3 visualize_dataset.py data.csv

# Train (auto-splits data.csv into train_set.csv + validation_set.csv)
python3 main.py data.csv
python3 main.py data.csv --epochs 100 --batch_size 32 --learning_rate 0.005 --layer 24 16 8 --adam

# Predict (uses export.json produced by training)
python3 predict.py validation_set.csv export.json
```

## Architecture

This is a NumPy-only MLP (no PyTorch/TensorFlow) for binary classification of breast cancer tumors (M/B) using the Wisconsin dataset (30 features, 569 samples).

**Training flow:**
1. `main.py` — parses args, calls `split_dataset()`, builds layer stack via `build_layers_from_args()`, trains via `NeuralNetwork.execute_training()`, serializes weights to `export.json`
2. `neural_network.py` — `NeuralNetwork` manages forward/backward passes and the training loop (mini-batch SGD with early stopping)
3. `layers.py` — `Layer` base class; subclasses: `Dense`, `ReLU`, `Sigmoid`, `Softmax`
4. `loss_functions.py` — `BinaryCrossEntropy` (inference) and `CategoricalCrossEntropy` (training)
5. `optimizers.py` — `SGD` and `Adam`
6. `utils.py` — CSV loading, z-score normalization, train/val split, shuffling
7. `predict.py` — loads `export.json`, performs a Softmax→Sigmoid conversion for inference, runs `fit_predict()`

**Training vs inference output heads:**
- Training uses a 2-neuron `Softmax` output + `CategoricalCrossEntropy`
- Inference converts the saved model to a 1-neuron `Sigmoid` + `BinaryCrossEntropy` by fusing the last Dense layer weights: `W_fused = W[:,0] - W[:,1]`

**Fused gradient optimization:**
`NeuralNetwork.configure_training()` sets `use_fused_gradient = True` on the output layer when Softmax+CCE or Sigmoid+BCE are paired. This skips the full Jacobian and uses the simplified delta `(ŷ - y) / batch_size` directly.

**Generated files** (not in git): `train_set.csv`, `validation_set.csv`, `export.json`