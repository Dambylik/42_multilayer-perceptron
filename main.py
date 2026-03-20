from neural_network import *
from loss_functions import CategoricalCrossEntropy
from optimizers import Adam, SGD
from utils import *
import json
import argparse
import sys


def build_layers_from_args(layer):
    if (layer is None):
        return (
        [
            Dense(30, 10),
            ReLU(),
            Dense(10, 24),
            ReLU(),
            Dense(24, 8),
            ReLU(),
            Dense(8, 2),
            Softmax()
        ])
    else:
        for neuron in layer:
            if (neuron <= 0):
                raise Exception("A layer must have at least 1 neuron")

        layers_final = []
        new_dense = Dense(30, layer[0])
        new_activation = ReLU()
        last_nb_neurons = layer[0]

        layers_final.append(new_dense)
        layers_final.append(new_activation)

        if (len(layer) == 1):
            new_dense = Dense(last_nb_neurons, 2)
            new_activation = Softmax()
            layers_final.append(new_dense)
            layers_final.append(new_activation)

        for i in range(1, len(layer)):
            # Final layer uses Softmax with 2 output neurons.
            if (i == len(layer) - 1):
                new_dense = Dense(last_nb_neurons, 2)
                new_activation = Softmax()
            else :
                new_dense = Dense(last_nb_neurons, layer[i])
                last_nb_neurons = layer[i]
                new_activation = ReLU()
            
            # Add Dense + activation pair.
            layers_final.append(new_dense)
            layers_final.append(new_activation)
        return (layers_final)


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP — pass a raw dataset to auto-split, or two pre-split files.",
        usage="python3 main.py data.csv [options]\n"
              "       python3 main.py train_set.csv validation_set.csv [options]"
    )
    parser.add_argument("file1", help="Raw dataset (auto-split) or train set")
    parser.add_argument("file2", nargs="?", default=None, help="Validation set (optional)")
    parser.add_argument("--split", type=float, default=0.2, help="Validation ratio when auto-splitting (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--layer", type=int, nargs="+")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--adam", action="store_true", help="Activate the Adam Optimizer")
    args = parser.parse_args()

    # Auto-split if only one file is given.
    if args.file2 is None:
        print(f"Auto-splitting '{args.file1}' ({int((1 - args.split) * 100)}/{int(args.split * 100)})...")
        try:
            split_dataset(args.file1, args.split)
        except (ValueError, FileNotFoundError) as err:
            print(f"Error: {err}")
            sys.exit(1)
        train_path, val_path = "train_set.csv", "validation_set.csv"
    else:
        train_path, val_path = args.file1, args.file2

    # Build layers from optional --layer argument.
    try:
        model = NeuralNetwork(build_layers_from_args(args.layer))
    except ValueError:
        print("A layer must contain at least 1 neuron")
        sys.exit(1)

    # Load train and validation datasets.
    # Normalization stats are derived from training data and reused on validation to prevent data leakage.
    try:
        X, y, train_means, train_stds = create_set_softmax(train_path)
        X_validation, y_validation, _, _ = create_set_softmax(val_path, train_means, train_stds)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return

    # Print dataset shapes.
    print(f"x_train shape : {X.shape}")
    print(f"x_valid shape : {X_validation.shape}")

    # Choose ADAM or SGD.
    if (args.adam):
        # ADAM / CCE
        model.configure_training(
            loss_criterion=CategoricalCrossEntropy(),
            weight_updater=Adam(build_layers_from_args(args.layer), lr=args.learning_rate)
        )
    else:
        # SGD / CCE
        model.configure_training(
            loss_criterion=CategoricalCrossEntropy(),
            weight_updater=SGD(lr=args.learning_rate)
        )

    # Train model.
    model.execute_training(X, y, X_validation, y_validation, epochs=args.epochs, batch_size=args.batch_size, early_stopping_patience=args.patience)

    # Export learned weights/biases and topology to JSON.
    export = []
    for layer in model.layers:
        name = type(layer).__name__
        if name == "Dense":
            export.append({
                "type": "Dense",
                "W": layer.weights.tolist(),
                "b": layer.biases.tolist()
            })
        else:
            export.append({"type": name})
    with open("export.json", "w") as f:
        json.dump(export, f, indent=4)
    print("saving model './export.json' to disk...")

if __name__ == "__main__":
    main()