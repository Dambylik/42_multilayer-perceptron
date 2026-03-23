import numpy as np

class Layer:
    counter = 0 # global counter to assign unique IDs

    def __init__(self):
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
        limit = np.sqrt(6 / input_size)
        matrix_shape = (input_size, neuron_count)
        self.weights = np.random.uniform(-limit, limit, size=matrix_shape)
        self.biases = np.zeros((1, neuron_count))

    def forward(self, input_data):
        self.stored_input = input_data          # save for backward pass
        weighted_sum = np.dot(input_data, self.weights)
        return weighted_sum + self.biases       # Z = X·W + b

    def backward(self, upstream_error):
        input_flipped = self.stored_input.T
        self.weight_gradients = np.dot(input_flipped, upstream_error)       # dW = X^T · δ
        self.bias_gradients = np.sum(upstream_error, axis=0, keepdims=True) # db = Σδ
        weights_flipped = self.weights.T
        error_to_pass_back = np.dot(upstream_error, weights_flipped)
        return error_to_pass_back

    def get_parameters(self):
        return [self.weights, self.biases]

    def get_gradients(self):
        return [self.weight_gradients, self.bias_gradients]


class ReLU(Layer):
    def forward(self, input_data):
        self.positive_input_mask = (input_data > 0)         # remember which neurons were 'on'
        return input_data * self.positive_input_mask        

    def backward(self, upstream_gradient):
        return upstream_gradient * self.positive_input_mask # block gradient where input was ≤ 0     


class Sigmoid(Layer):
    def forward(self, input_data):
        self.activated_output = 1 / (1 + np.exp(-input_data))
        return self.activated_output

    def backward(self, upstream_gradient):
        local_derivative = self.activated_output * (1 - self.activated_output)
        return upstream_gradient * local_derivative

    def backward_last_layer(self, upstream_gradient): 
        return upstream_gradient                            # fused gradient shortcut
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        stable_input = input_data - np.max(input_data, axis=1, keepdims=True) # numerical stability
        unnormalized_probs = np.exp(stable_input)
        self.probability_distribution = unnormalized_probs / np.sum(unnormalized_probs, axis=1, keepdims=True)
        return self.probability_distribution

    def backward_last_layer(self, upstream_gradient): 
        return upstream_gradient                            # fused gradient shortcut



import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 3))

axes[0].plot(x, np.maximum(0, x), color='royalblue')
axes[0].set_title('ReLU: max(0, x)')
axes[0].set_ylabel('Output')
axes[0].spines['left'].set_position('zero')
axes[0].spines['bottom'].set_position('zero')
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)

axes[1].plot(x, 1 / (1 + np.exp(-x)), color='tomato')
axes[1].set_title('Sigmoid: 1 / (1 + e⁻ˣ)')
axes[1].set_ylabel('Output')
axes[1].spines['left'].set_position('zero')
axes[1].spines['bottom'].set_position('zero')
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)

z = np.array([[1.0, 2.0, 0.5]])  # example logits
e = np.exp(z - z.max())
s = (e / e.sum()).flatten()
axes[2].bar(['Class A', 'Class B', 'Class C'], s, color='mediumseagreen')
axes[2].set_title(f'Softmax output (sums to {s.sum():.1f})')
axes[2].set_ylabel('Probability')

for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/relu_sig_soft_loss.png", dpi=150)
print("Saved images/relu_sig_soft_loss.png")
plt.show()