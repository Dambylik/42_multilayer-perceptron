import numpy as np

class Layer:
    # Tracking how many layer objects exist for unique ID assignment
    counter = 0 

    def __init__(self):
        # Metadata to help the Sequential model manage the flow
        self.is_output_layer = False
        self.use_fused_gradient = False
        self.layer_id = Layer.counter
        Layer.counter += 1

    def forward(self, input_data):
        """Pass data toward the output."""
        raise NotImplementedError

    def backward(self, upstream_gradient):
        """Pass the error signal back toward the input."""
        raise NotImplementedError

    def get_parameters(self):
        """Return weights and biases if they exist."""
        return []

    def get_gradients(self):
        """Return dW and db calculated during backward pass."""
        return []


class Dense(Layer):
    def __init__(self, input_size, neuron_count):
        super().__init__()
        
        # Xavier Initialization: Finding the mathematical "sweet spot"
        limit = np.sqrt(6 / input_size)
        matrix_shape = (input_size, neuron_count)
        
        # Initializing weights and biases
        self.weights = np.random.uniform(-limit, limit, size=matrix_shape)
        self.biases = np.zeros((1, neuron_count))

    def forward(self, input_data):
        # Store a copy for the backward pass
        self.stored_input = input_data
        
        # Calculate: (Data * Weights) + Biases
        weighted_sum = np.dot(input_data, self.weights)
        return weighted_sum + self.biases

    def backward(self, upstream_error):
        # 1. Calculate how much weights should change
        # We must transpose (flip) the input to align the dimensions
        input_flipped = self.stored_input.T
        self.weight_gradients = np.dot(input_flipped, upstream_error)
        
        # 2. Calculate how much biases should change
        # We sum the errors across the batch (axis=0)
        self.bias_gradients = np.sum(upstream_error, axis=0, keepdims=True)
        
        # 3. Calculate error to send to the PREVIOUS layer
        # We flip the weights to "reverse" the flow of information
        weights_flipped = self.weights.T
        error_to_pass_back = np.dot(upstream_error, weights_flipped)
        
        return error_to_pass_back

    def get_parameters(self):
        return [self.weights, self.biases]

    def get_gradients(self):
        return [self.weight_gradients, self.bias_gradients]


class ReLU(Layer):
    def forward(self, input_data):
        # The Mask: A list of True/False (True if value > 0)
        self.positive_input_mask = (input_data > 0)
        
        # Logic: If negative, it becomes 0. If positive, it stays the same.
        return input_data * self.positive_input_mask

    def backward(self, upstream_gradient):
        # Logic: The error only flows back through neurons that were "On" (positive).
        # If a neuron was "Off" (0), the gradient is blocked.
        return upstream_gradient * self.positive_input_mask

class Sigmoid(Layer):
    def forward(self, input_data):
        # Formula: 1 / (1 + e^-x)
        self.activated_output = 1 / (1 + np.exp(-input_data))
        return self.activated_output

    def backward(self, upstream_gradient):
        # The derivative of Sigmoid is: output * (1 - output)
        local_derivative = self.activated_output * (1 - self.activated_output)
        return upstream_gradient * local_derivative

    def backward_last_layer(self, upstream_gradient): 
        # The Shortcut: If Sigmoid is the very last layer, 
        # we skip the complex derivative for speed (Optimization).
        return upstream_gradient
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # Numerical Stability Trick: Subtract the max value 
        # This prevents "inf" (infinity) errors when calculating exp().
        stable_input = input_data - np.max(input_data, axis=1, keepdims=True)
        
        unnormalized_probs = np.exp(stable_input)
        
        # Normalize: Divide each value by the sum of all values in the row.
        self.probability_distribution = unnormalized_probs / np.sum(unnormalized_probs, axis=1, keepdims=True)
        return self.probability_distribution

    def backward_last_layer(self, upstream_gradient): 
        # Like Sigmoid, we use the "Fused" gradient shortcut here.
        return upstream_gradient
