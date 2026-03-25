import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss
    """

    def __init__(self):
        pass

    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.y_true = truth
        term_1 = truth * np.log(self.y_pred)
        term_2 = (1 - truth) * np.log(1 - self.y_pred)
        return -np.mean(term_1 + term_2)

    def backward(self):
        # Fused gradient: δ = (ŷ - y) / N
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss
    """

    def __init__(self):
        pass

    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)   # avoid log(0)
        self.y_true = truth
        return -np.mean(np.sum(truth * np.log(self.y_pred), axis=1))

    def backward(self):
        # Fused gradient: δ = (ŷ - y) / N
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]
