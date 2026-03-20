import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
'''weights = [0.2, 0.8, -0.5, 1.0]'''
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biasis = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biasis2 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biasis
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biasis2
print(layer1_output)
print(layer2_output)

'''weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]'''

'''biasis = [2, 3, 0.5]'''

'''
layer_outputs = [] #output of the current layer
for neuron_weights, neuron_bias in zip(weights, biasis):
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs) '''