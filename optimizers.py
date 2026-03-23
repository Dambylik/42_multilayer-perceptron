import numpy as np
from layers import Dense, Layer


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, parameters, gradients, layer=None):
        for p, g in zip(parameters, gradients):
            p -= self.lr * g                    # Weight = Weight - (Speed * Error)


class Adam:
    def __init__(self, layers, lr=0.001, first_moment_decay=0.9, second_moment_decay=0.999, numerical_stability_constant=1e-8):
        # Hyperparameters
        self.learning_rate = lr
        self.first_moment_decay = first_moment_decay   # beta1
        self.second_moment_decay = second_moment_decay # beta2
        self.epsilon = numerical_stability_constant

        # State tracking (moving averages of gradients)
        self.first_moments_weights = []  # mW: Mean of gradients
        self.first_moments_biases = []   # mb
        self.second_moments_weights = [] # vW: Variance of gradients
        self.second_moments_biases = []  # vb
        self.iteration_step = 1          # t

        # Initialize state tensors for each layer in the network
        for layer in layers:
            if isinstance(layer, Dense):
                # We need a history buffer matching the shape of weights and biases
                rows, cols = layer.weights.shape
                self.first_moments_weights.append(np.zeros((rows, cols)))
                self.first_moments_biases.append(np.zeros((1, cols)))
                self.second_moments_weights.append(np.zeros((rows, cols)))
                self.second_moments_biases.append(np.zeros((1, cols)))
            else:
                # Placeholder for non-parameter layers (like ReLU)
                self.first_moments_weights.append(None)
                self.first_moments_biases.append(None)
                self.second_moments_weights.append(None)
                self.second_moments_biases.append(None)

    def step(self, parameter_list, gradient_list, layer):
        # Adam only updates layers with learnable parameters (Dense)
        if not isinstance(layer, Dense):
            return

        layer_idx = layer.layer_id
        
        # We loop through the parameters (Weights first, then Biases)
        for i, (param_tensor, grad_tensor) in enumerate(zip(parameter_list, gradient_list)):
            
            if i == 0: # Processing Weights
                # 1. Update First Moment (Moving Average of Gradients)
                self.first_moments_weights[layer_idx] = (self.first_moment_decay * self.first_moments_weights[layer_idx] + 
                                                         (1 - self.first_moment_decay) * grad_tensor)
                
                # 2. Update Second Moment (Moving Average of Squared Gradients)
                self.second_moments_weights[layer_idx] = (self.second_moment_decay * self.second_moments_weights[layer_idx] + 
                                                          (1 - self.second_moment_decay) * (grad_tensor ** 2))

                # 3. Bias Correction (Accounts for the moments being 0 at the start)
                corrected_first_moment = self.first_moments_weights[layer_idx] / (1 - self.first_moment_decay ** self.iteration_step)
                corrected_second_moment = self.second_moments_weights[layer_idx] / (1 - self.second_moment_decay ** self.iteration_step)

                # 4. Update the actual Weight values
                param_tensor -= self.learning_rate * corrected_first_moment / (np.sqrt(corrected_second_moment) + self.epsilon)

            else: # Processing Biases
                self.first_moments_biases[layer_idx] = (self.first_moment_decay * self.first_moments_biases[layer_idx] + 
                                                        (1 - self.first_moment_decay) * grad_tensor)
                
                self.second_moments_biases[layer_idx] = (self.second_moment_decay * self.second_moments_biases[layer_idx] + 
                                                         (1 - self.second_moment_decay) * (grad_tensor ** 2))

                corrected_first_moment = self.first_moments_biases[layer_idx] / (1 - self.first_moment_decay ** self.iteration_step)
                corrected_second_moment = self.second_moments_biases[layer_idx] / (1 - self.second_moment_decay ** self.iteration_step)

                param_tensor -= self.learning_rate * corrected_first_moment / (np.sqrt(corrected_second_moment) + self.epsilon)

        # Increment time step after a full layer update
        if layer_idx == Layer.counter - 1:
            self.iteration_step += 1