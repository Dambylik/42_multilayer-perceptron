import numpy as np
from loss_functions import BinaryCrossEntropy, CategoricalCrossEntropy
from layers import *
from utils import shuffle_data


class NeuralNetwork:
    def __init__(self, layers_stack):
        self.layers = layers_stack

    def configure_training(self, loss_criterion, weight_updater):
        """Sets the loss and optimizer rules."""
        self.criterion = loss_criterion
        self.optimizer = weight_updater

        # Set metadata for the very last layer
        output_layer = self.layers[-1]
        output_layer.is_output_layer = True

        # Optimization: Use 'fused' gradients for common combinations
        if isinstance(self.criterion, BinaryCrossEntropy) and isinstance(output_layer, Sigmoid):
            output_layer.use_fused_gradient = True
        elif isinstance(self.criterion, CategoricalCrossEntropy) and isinstance(output_layer, Softmax):
            output_layer.use_fused_gradient = True

    def propagate_forward(self, input_data):
        """Passes data through every layer to get a prediction."""
        current_flow = input_data
        for layer in self.layers:
            current_flow = layer.forward(current_flow)
        return current_flow

    def propagate_backward(self, loss_gradient):
        """Sends the error signal backward through the layers."""
        current_gradient = loss_gradient
        for layer in reversed(self.layers):
            if layer.use_fused_gradient:
                current_gradient = layer.backward_last_layer(current_gradient)
            else:
                current_gradient = layer.backward(current_gradient)

    def execute_training(self, train_X, train_y, val_X, val_y, epochs=100, batch_size=32, early_stopping_patience=3):
        """The main loop that teaches the model."""
        from visualize_graphs import show_combined_graph
        num_train_samples = train_X.shape[0]

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        # Early stopping logic
        best_val_loss = float("inf")
        patience_counter = 0
        epochs_run = 0

        for epoch in range(epochs):
            # 1. Shuffle data to prevent the model from 'memorizing' row order
            train_X, train_y = shuffle_data(train_X, train_y)
            epoch_total_loss = 0
            epoch_correct = 0

            # 2. Batch Loop: Process small groups of patients
            for start in range(0, num_train_samples, batch_size):
                end = start + batch_size
                batch_X, batch_y = train_X[start:end], train_y[start:end]

                # --- STEP A: Forward Pass ---
                predictions = self.propagate_forward(batch_X)

                # --- STEP B: Calculate Loss ---
                batch_loss = self.criterion.forward(predictions, batch_y)
                epoch_total_loss += batch_loss * batch_X.shape[0]

                # Training accuracy
                epoch_correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_y, axis=1))

                # --- STEP C: Backward Pass ---
                error_signal = self.criterion.backward()
                self.propagate_backward(error_signal)

                # --- STEP D: Update Weights ---
                for layer in self.layers:
                    self.optimizer.step(layer.get_parameters(), layer.get_gradients(), layer)

            # 3. Validation Phase: Check performance on unseen data
            val_loss, val_acc = self.evaluate_performance(val_X, val_y)

            avg_train_loss = epoch_total_loss / num_train_samples
            avg_train_acc = epoch_correct / num_train_samples

            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(avg_train_acc)
            val_acc_history.append(val_acc)

            epochs_run += 1

            # 4. Logging & Early Stopping
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {avg_train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered. Model has stopped improving.")
                break

        show_combined_graph(epochs_run, train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    def fit_predict(self, X_validation, y_validation, batch_size=2, shuffle=False):
        """Forward-only inference: shows confusion matrix and reports loss + accuracy."""
        import matplotlib.pyplot as plt

        n_samples = X_validation.shape[0]
        if shuffle:
            X_validation, y_validation = shuffle_data(X_validation, y_validation)

        total_loss = 0.0
        all_preds = []
        all_true = []

        for start in range(0, n_samples, batch_size):
            X_batch = X_validation[start:start + batch_size]
            y_batch = y_validation[start:start + batch_size]

            y_pred = self.propagate_forward(X_batch)

            total_loss += self.criterion.forward(y_pred, y_batch) * X_batch.shape[0]
            all_preds.extend((y_pred >= 0.5).astype(int).flatten().tolist())
            all_true.extend(y_batch.flatten().tolist())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        loss = total_loss / n_samples
        accuracy = np.mean(all_preds == all_true)
        print(f"Loss: {loss:.4f} - accuracy: {accuracy:.4f}")

        # Confusion matrix: rows = actual, cols = predicted
        # Labels: 0 = B (benign), 1 = M (malignant)
        classes = ["B (benign)", "M (malignant)"]
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(all_true.astype(int), all_preds.astype(int)):
            cm[t][p] += 1

        _, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix  —  loss: {loss:.4f}  acc: {accuracy:.4f}")

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i][j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]}",
                        ha="center", va="center", color=color, fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig("images/confusion_matrix.png", dpi=150)
        print("Saved images/confusion_matrix.png")
        plt.show()

    def evaluate_performance(self, features, labels):
        """Internal helper to calculate loss and accuracy."""
        predictions = self.propagate_forward(features)
        loss = self.criterion.forward(predictions, labels)
        
        # Accuracy logic for 2-neuron Softmax output
        pred_indices = np.argmax(predictions, axis=1)
        true_indices = np.argmax(labels, axis=1)
        accuracy = np.mean(pred_indices == true_indices)
        
        return loss, accuracy