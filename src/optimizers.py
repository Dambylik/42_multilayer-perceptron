import numpy as np
from src.layers import Dense, Layer


class SGD:
    """
    Stochastic Gradient Descent.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, parameters, gradients, layer=None):
        for p, g in zip(parameters, gradients):
            p -= self.lr * g


class Adam:
    """
    Adaptive Moment Estimation 
    """
    def __init__(self, layers, lr=0.001,
                 first_moment_decay=0.9,
                 second_moment_decay=0.999,
                 numerical_stability_constant=1e-8):

        self.learning_rate         = lr
        self.first_moment_decay    = first_moment_decay    # β₁
        self.second_moment_decay   = second_moment_decay   # β₂
        self.epsilon               = numerical_stability_constant
        self.iteration_step        = 1                     # t

        self.first_moments_weights  = []  # mW
        self.first_moments_biases   = []  # mb
        self.second_moments_weights = []  # vW
        self.second_moments_biases  = []  # vb

        n_dense = 0
        for layer in layers:
            if isinstance(layer, Dense):
                rows, cols = layer.weights.shape
                self.first_moments_weights.append(np.zeros((rows, cols)))
                self.first_moments_biases.append(np.zeros((1, cols)))
                self.second_moments_weights.append(np.zeros((rows, cols)))
                self.second_moments_biases.append(np.zeros((1, cols)))
                n_dense += 1
            else:
                self.first_moments_weights.append(None)
                self.first_moments_biases.append(None)
                self.second_moments_weights.append(None)
                self.second_moments_biases.append(None)



    def step(self, parameter_list, gradient_list, layer):
        if not isinstance(layer, Dense):
            return

        idx = layer.layer_id

        for i, (param, grad) in enumerate(zip(parameter_list, gradient_list)):

            if i == 0:  # ── Weights ──────────────────────────────────────────
                # 1. Update first moment (moving average of gradient direction)
                self.first_moments_weights[idx] = (
                    self.first_moment_decay * self.first_moments_weights[idx]
                    + (1 - self.first_moment_decay) * grad)

                # 2. Update second moment (moving average of gradient magnitude²)
                self.second_moments_weights[idx] = (
                    self.second_moment_decay * self.second_moments_weights[idx]
                    + (1 - self.second_moment_decay) * (grad ** 2))

                # 3. Bias correction (moments start at 0, correction inflates them early on)
                m_hat = self.first_moments_weights[idx]  / (1 - self.first_moment_decay  ** self.iteration_step)
                v_hat = self.second_moments_weights[idx] / (1 - self.second_moment_decay ** self.iteration_step)

                # 4. Apply update
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            else:    # ── Biases ───────────────────────────────────────────
                self.first_moments_biases[idx] = (
                    self.first_moment_decay * self.first_moments_biases[idx]
                    + (1 - self.first_moment_decay) * grad)

                self.second_moments_biases[idx] = (
                    self.second_moment_decay * self.second_moments_biases[idx]
                    + (1 - self.second_moment_decay) * (grad ** 2))

                m_hat = self.first_moments_biases[idx]  / (1 - self.first_moment_decay  ** self.iteration_step)
                v_hat = self.second_moments_biases[idx] / (1 - self.second_moment_decay ** self.iteration_step)

                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Increment global time step once per full network pass
        if idx == Layer.counter - 1:
            self.iteration_step += 1
