import numpy as np


class BinaryCrossEntropy:
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
    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.y_true = truth
        return -np.mean(np.sum(truth * np.log(self.y_pred), axis=1))

    def backward(self):
        # Fused gradient: δ = (ŷ - y) / N
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


# Visualize: how loss grows as prediction gets worse
import matplotlib.pyplot as plt

p = np.linspace(0.01, 0.99, 200)
plt.figure(figsize=(7, 4))
plt.plot(p, -np.log(p),     label='BCE when y=1  →  −log(ŷ)',       color='tomato')
plt.plot(p, -np.log(1 - p), label='BCE when y=0  →  −log(1−ŷ)',     color='royalblue')
plt.xlabel('Predicted probability ŷ')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy: loss explodes when prediction is wrong')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/bce_loss.png", dpi=150)
print("Saved images/bce_loss.png")
plt.show()