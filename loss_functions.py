import numpy as np

class BinaryCrossEntropy:
    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.y_true = truth

        term_1 = truth * np.log(self.y_pred)
        term_2 = (1 - truth) * np.log(1 - self.y_pred)
        return -np.mean(term_1 + term_2)

    def backward(self):
        # Fused gradient shortcut for BCE + Sigmoid last layer
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


class CategoricalCrossEntropy:
    def forward(self, prediction, truth):
        self.y_pred = np.clip(prediction, 1e-7, 1 - 1e-7)
        self.y_true = truth

        return -np.mean(np.sum(truth * np.log(self.y_pred), axis=1))

    def backward(self):
        # Fused gradient shortcut for CCE + Softmax last layer
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


class MeanSquaredError:
    def forward(self, predicted_values, actual_values):
        # Store for the backward pass
        self.predictions = predicted_values
        self.targets = actual_values
        
        # Calculate the average of the squared differences
        # Formula: mean((pred - target)^2)
        differences = predicted_values - actual_values
        squared_errors = np.square(differences)
        return np.mean(squared_errors)

    def backward(self):
        # The derivative of (p - t)^2 with respect to p is 2 * (p - t)
        # We divide by the total number of elements to match the 'mean' in forward
        num_elements = self.predictions.size
        error_signal = 2 * (self.predictions - self.targets) / num_elements
        return error_signal