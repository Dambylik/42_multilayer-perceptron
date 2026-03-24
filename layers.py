import numpy as np


class Layer:
    """
    Abstract base class for every layer in the network.

    All layers share:
      - a unique layer_id  (used by Adam to index its moment buffers)
      - is_output_layer    (set True by NeuralNetMLP for the last layer)
      - use_fused_gradient (set True when Softmax+CCE or Sigmoid+BCE are paired)
    """

    counter = 0  # global counter — gives each layer a unique ID

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


# ──────────────────────────────────────────────────────────────────────────────

class Dense(Layer):
    """
    Fully-connected layer: every input neuron connects to every output neuron.

    Forward  :  Z = X · W + b
    Backward :  dW = Xᵀ · δ     db = Σδ     δ_prev = δ · Wᵀ

    Weight init: LeCun uniform  W ~ Uniform(-√(6/n_in), +√(6/n_in))
    Bias   init: zeros
    """

    def __init__(self, input_size, neuron_count):
        super().__init__()
        limit = np.sqrt(6 / input_size)
        self.weights = np.random.uniform(-limit, limit, size=(input_size, neuron_count))
        self.biases  = np.zeros((1, neuron_count))

        # print(f"    [Layer {self.layer_id}] Dense  "
        #       f"({input_size:3d} → {neuron_count:3d})  "
        #       f"W: {self.weights.shape}  "
        #       f"init range: ±{limit:.4f}  "
        #       f"params: {input_size*neuron_count + neuron_count}")

    def forward(self, input_data):
        self.stored_input = input_data                           # save for backward
        return np.dot(input_data, self.weights) + self.biases   # Z = X·W + b

    def backward(self, upstream_error):
        self.weight_gradients = np.dot(self.stored_input.T, upstream_error)        # dW = Xᵀ·δ
        self.bias_gradients   = np.sum(upstream_error, axis=0, keepdims=True)      # db = Σδ
        return np.dot(upstream_error, self.weights.T)                               # δ_prev = δ·Wᵀ

    def get_parameters(self):
        return [self.weights, self.biases]

    def get_gradients(self):
        return [self.weight_gradients, self.bias_gradients]


# ──────────────────────────────────────────────────────────────────────────────

class ReLU(Layer):
    """
    Rectified Linear Unit — hidden-layer activation.

    Forward  :  a = max(0, z)
    Backward :  δ_prev = δ * (z > 0)   ← only passes gradient where input was positive

    Why ReLU?
      - Computationally trivial (just a threshold)
      - Does not saturate for large positive inputs (unlike Sigmoid)
      - Introduces non-linearity so the network can learn curved boundaries
      - Dead-neuron risk: if z ≤ 0 for all samples, the neuron never updates
    """

    def forward(self, input_data):
        self.positive_input_mask = (input_data > 0)      # remember which were 'on'
        return input_data * self.positive_input_mask      # kill all negatives

    def backward(self, upstream_gradient):
        return upstream_gradient * self.positive_input_mask  # block gradient where z ≤ 0


# ──────────────────────────────────────────────────────────────────────────────

class Sigmoid(Layer):
    """
    Sigmoid — used as the output layer during *inference* only.

    Forward  :  σ(z) = 1 / (1 + e^{-z})   output ∈ (0, 1)  → P(Malignant)
    Backward :  δ_prev = δ * σ(z) * (1 − σ(z))

    Fused shortcut (backward_last_layer):
      When paired with BinaryCrossEntropy the full derivative simplifies to
      δ = (ŷ − y) / N  — computed in the loss, just passed through here.
    """

    def forward(self, input_data):
        self.activated_output = 1 / (1 + np.exp(-input_data))
        return self.activated_output

    def backward(self, upstream_gradient):
        local_derivative = self.activated_output * (1 - self.activated_output)
        return upstream_gradient * local_derivative

    def backward_last_layer(self, upstream_gradient):
        return upstream_gradient   # fused gradient: loss already did the hard work


# ──────────────────────────────────────────────────────────────────────────────

class Softmax(Layer):
    """
    Softmax — output layer during *training*.

    Forward:
      stable_z = z - max(z)           ← subtract max for numerical stability
      ŷ = exp(stable_z) / Σexp(...)   ← normalise into a probability distribution

    Output: vector of probabilities summing to 1.
      ŷ[0] = P(Malignant)   ŷ[1] = P(Benign)

    Fused shortcut (backward_last_layer):
      When paired with CategoricalCrossEntropy the Softmax Jacobian and CCE
      gradient collapse to the beautifully simple:
        δ = (ŷ − y_true) / batch_size
      This is exact — not an approximation.
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
        return upstream_gradient   # fused gradient: loss already computed δ = (ŷ−y)/N
