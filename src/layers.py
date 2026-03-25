import numpy as np


class Layer:
    """
    Abstract base class for every layer in the network.
    """

    counter = 0

    def __init__(self):
        self.is_output_layer    = False
        self.use_fused_gradient = False
        self.layer_id           = Layer.counter
        Layer.counter          += 1

    def forward(self, input_data):
        """Pass data toward the output."""
        raise NotImplementedError

    def backward(self, upstream_gradient):
        """Pass the error signal back toward the input."""
        raise NotImplementedError

    def get_parameters(self):
        """Return [weights, biases] for layers that have learnable params."""
        return []

    def get_gradients(self):
        """Return [dW, db] computed during the backward pass."""
        return []


class Dense(Layer):
    """
    Fully-connected layer: every input neuron connects to every output neuron.
    """

    def __init__(self, input_size, neuron_count):
        super().__init__()
        limit = np.sqrt(6 / input_size)
        self.weights = np.random.uniform(-limit, limit, size=(input_size, neuron_count))
        self.biases  = np.zeros((1, neuron_count))
        

    def forward(self, input_data):
        self.stored_input = input_data                          # save for backward
        return np.dot(input_data, self.weights) + self.biases   # Z = X·W + b


    def backward(self, upstream_error):
        self.weight_gradients = np.dot(self.stored_input.T, upstream_error)        # dW = Xᵀ·δ
        self.bias_gradients   = np.sum(upstream_error, axis=0, keepdims=True)      # db = Σδ
        return np.dot(upstream_error, self.weights.T)                              # δ_prev = δ·Wᵀ


    def get_parameters(self):
        return [self.weights, self.biases]
    
    
    def get_gradients(self):
        return [self.weight_gradients, self.bias_gradients]


class ReLU(Layer):
    """
    Rectified Linear Unit — hidden-layer activation.
    """

    def forward(self, input_data):
        self.positive_input_mask = (input_data > 0)      # remember which were 'on'
        return input_data * self.positive_input_mask     # kill all negatives


    def backward(self, upstream_gradient):
        return upstream_gradient * self.positive_input_mask  # block gradient where z ≤ 0



class Softmax(Layer):
    """
    Softmax — output layer during training.
    """

    def __init__(self):
        super().__init__()


    def forward(self, input_data):
        stable_input              = input_data - np.max(input_data, axis=1, keepdims=True)
        unnormalized_probs        = np.exp(stable_input)
        self.probability_distribution = unnormalized_probs / np.sum(unnormalized_probs,
                                                                      axis=1, keepdims=True)
        return self.probability_distribution


    def backward_last_layer(self, upstream_gradient):
        return upstream_gradient 
