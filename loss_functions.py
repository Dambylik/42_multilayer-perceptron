import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss — used during *inference* with a Sigmoid output.

    Formula:
        L = -mean( y·log(ŷ) + (1-y)·log(1-ŷ) )

    Intuition:
        y=1 (Malignant): loss = -log(ŷ)   → huge penalty if ŷ ≈ 0
        y=0 (Benign):    loss = -log(1-ŷ) → huge penalty if ŷ ≈ 1

    Fused backward (paired with Sigmoid):
        δ = (ŷ - y) / N      ← simplified from the full chain rule
    """

    def __init__(self):
        print(f"    [Loss] BinaryCrossEntropy")
        print(f"           L = -mean( y·log(ŷ) + (1-y)·log(1-ŷ) )")
        print(f"           Used with: Sigmoid output (inference)")

    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)   # avoid log(0)
        self.y_true = truth
        term_1 = truth * np.log(self.y_pred)
        term_2 = (1 - truth) * np.log(1 - self.y_pred)
        return -np.mean(term_1 + term_2)

    def backward(self):
        # Fused gradient: δ = (ŷ - y) / N
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss — used during *training* with a Softmax output.

    Formula:
        L = -mean( Σ_c  y_c · log(ŷ_c) )

    Intuition:
        Only the term for the TRUE class contributes (all others are multiplied by 0).
        So for a Malignant sample: L = -log(ŷ[M])
        The closer ŷ[M] is to 1.0, the smaller the loss.

    Fused backward (paired with Softmax):
        δ = (ŷ - y_true) / N      ← simplified from Softmax Jacobian + CCE gradient
        This is mathematically exact — not an approximation.
    """

    def __init__(self):
        print(f"    [Loss] CategoricalCrossEntropy")
        print(f"           L = -mean( Σ_c  y_c · log(ŷ_c) )")
        print(f"           Used with: Softmax output (training)")

    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)   # avoid log(0)
        self.y_true = truth
        return -np.mean(np.sum(truth * np.log(self.y_pred), axis=1))

    def backward(self):
        # Fused gradient: δ = (ŷ - y) / N
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]
