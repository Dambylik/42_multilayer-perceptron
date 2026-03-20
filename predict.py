import json
import numpy as np
import argparse
from layers import Dense, ReLU, Sigmoid, Softmax
from neural_network import NeuralNetwork
from loss_functions import BinaryCrossEntropy
from optimizers import SGD
from utils import create_set_sigmoid

def reconstruct_model_from_json(serialized_data):
    """Restores the layer objects using the saved weight matrices."""
    layers_list = []
    activation_map = {"ReLU": ReLU, "Sigmoid": Sigmoid, "Softmax": Softmax}
    
    for layer_config in serialized_data:
        layer_type = layer_config["type"]
        
        if layer_type == "Dense":
            # Extract dimensions from the saved Weight matrix
            input_dim = len(layer_config["W"])
            output_dim = len(layer_config["W"][0])
            
            new_layer = Dense(input_dim, output_dim)
            new_layer.weights = np.array(layer_config["W"])
            new_layer.biases = np.array(layer_config["b"])
            layers_list.append(new_layer)
        elif layer_type in activation_map:
            layers_list.append(activation_map[layer_type]())
            
    return layers_list

def run_inference():
    # Program setup: Handling file paths
    cli_parser = argparse.ArgumentParser(description="MLP Prediction Utility")
    cli_parser.add_argument("data_file", nargs="?", default="validation_set.csv")
    cli_parser.add_argument("model_file", nargs="?", default="export.json")
    args = cli_parser.parse_args()

    try:
        # 1. Load the "DNA" of the trained model
        with open(args.model_file, "r") as json_file:
            reconstructed_layers = reconstruct_model_from_json(json.load(json_file))

        # 2. Load the validation features and true labels
        val_features, val_labels, _, _ = create_set_sigmoid(args.data_file)
    except Exception as error:
        print(f"File loading failed: {error}")
        return

    # --- THE TRANSFORMATION NOTION ---
    # We convert a 2-neuron Softmax (Class A vs Class B) 
    # into a 1-neuron Sigmoid (Probability of Class A).
    
    # 3. Swap the last activation
    reconstructed_layers[-1] = Sigmoid()

    # 4. Fuse the weights: Subtracting neuron_2 weights from neuron_1 weights
    # Mathematically: Sigmoid(Z1 - Z2) is identical to Softmax(Z1, Z2)
    last_dense = reconstructed_layers[-2]
    fused_weights = (last_dense.weights[:, 0] - last_dense.weights[:, 1]).reshape(-1, 1)
    fused_bias = last_dense.biases[0][0] - last_dense.biases[0][1]

    # Update the layer with the new "Binary" parameters
    last_dense.weights = fused_weights
    last_dense.biases = np.array([[fused_bias]])

    # 5. Build the inference engine
    inference_model = NeuralNetwork(reconstructed_layers)
    inference_model.configure_training(
        loss_criterion=BinaryCrossEntropy(),
        weight_updater=SGD(lr=0.1)
    )

    # 6. Execute predictions
    inference_model.fit_predict(val_features, val_labels, batch_size=4, shuffle=False)

if __name__ == "__main__":
    run_inference()