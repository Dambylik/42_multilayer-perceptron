import numpy as np
import matplotlib.pyplot as plt
from loss_functions import BinaryCrossEntropy, CategoricalCrossEntropy
from layers import Sigmoid, Softmax
from utils import shuffle_data
from visualize_graphs import show_combined_graph


class NeuralNetMLP:
    def __init__(self, layers):
        self.layers = layers

    def configure_training(self, loss_criterion, weight_updater):
        self.criterion = loss_criterion
        self.optimizer = weight_updater

        output_layer = self.layers[-1]
        output_layer.is_output_layer = True

        # Fused gradient: skip full Jacobian for Softmax+CCE and Sigmoid+BCE
        if (isinstance(self.criterion, CategoricalCrossEntropy) and isinstance(output_layer, Softmax)) or \
           (isinstance(self.criterion, BinaryCrossEntropy) and isinstance(output_layer, Sigmoid)):
            output_layer.use_fused_gradient = True

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            if layer.use_fused_gradient:
                grad = layer.backward_last_layer(grad)
            else:
                grad = layer.backward(grad)

    def _update_weights(self):
        for layer in self.layers:
            self.optimizer.step(layer.get_parameters(), layer.get_gradients(), layer)

    def execute_training(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, early_stopping_patience=3):
        n = X_train.shape[0]
        train_loss_hist, val_loss_hist = [], []
        train_acc_hist, val_acc_hist = [], []

        best_val_loss = float("inf")
        patience_counter = 0
        epochs_run = 0

        for epoch in range(epochs):
            X_train, y_train = shuffle_data(X_train, y_train)
            total_loss = 0
            correct = 0

            for start in range(0, n, batch_size):
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                preds = self.forward(X_batch)
                total_loss += self.criterion.forward(preds, y_batch) * X_batch.shape[0]
                correct += np.sum(np.argmax(preds, axis=1) == np.argmax(y_batch, axis=1))

                self.backward(self.criterion.backward())
                self._update_weights()

            val_loss, val_acc = self.evaluate(X_val, y_val)
            train_loss = total_loss / n
            train_acc = correct / n

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            epochs_run += 1

            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        show_combined_graph(epochs_run, train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)

    def fit_predict(self, X_val, y_val, batch_size=2, shuffle=False):
        """Run inference and display a confusion matrix."""
        n = X_val.shape[0]
        if shuffle:
            X_val, y_val = shuffle_data(X_val, y_val)

        total_loss = 0.0
        all_preds, all_true = [], []

        for start in range(0, n, batch_size):
            X_batch = X_val[start:start + batch_size]
            y_batch = y_val[start:start + batch_size]
            y_pred = self.forward(X_batch)
            total_loss += self.criterion.forward(y_pred, y_batch) * X_batch.shape[0]
            all_preds.extend((y_pred >= 0.5).astype(int).flatten().tolist())
            all_true.extend(y_batch.flatten().tolist())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        loss = total_loss / n
        accuracy = np.mean(all_preds == all_true)
        print(f"Loss: {loss:.4f} - accuracy: {accuracy:.4f}")

        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(all_true.astype(int), all_preds.astype(int)):
            cm[t][p] += 1

        classes = ["B (benign)", "M (malignant)"]
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

    def evaluate(self, X, y):
        preds = self.forward(X)
        loss = self.criterion.forward(preds, y)
        accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy
