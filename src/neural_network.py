import numpy as np
from src.loss_functions import BinaryCrossEntropy, CategoricalCrossEntropy
from src.layers import Softmax
from tools.utils import shuffle_data, section, subsection
from tools.visualize_graphs import show_combined_graph, show_confusion_matrix


class NeuralNetMLP:
    """
    Multilayer Perceptron — manages the full forward / backward / update cycle.

    Responsibilities:
        configure_training()  — attach a loss and an optimizer, enable fused gradient
        forward()             — pass a batch through every layer in order
        backward()            — pass the error signal back through every layer
        update_weights()     — ask the optimizer to update each layer's parameters
        execute_training()    — full training loop with early stopping
        fit_predict()         — inference loop + confusion matrix
        evaluate()            — loss + accuracy on a full dataset (no update)
    """

    def __init__(self, layers):
        self.layers = layers


    def configure_training(self, loss_criterion, weight_updater):
        """
        Attach a loss function and an optimizer to the network.
        """
        self.criterion = loss_criterion
        self.optimizer = weight_updater

        output_layer = self.layers[-1]
        output_layer.is_output_layer = True

        output_layer.use_fused_gradient = (
            isinstance(self.criterion, CategoricalCrossEntropy)
            and isinstance(output_layer, Softmax)
        )


    def forward(self, X):
        """Run X through every layer in order and return the final predictions."""
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def backward(self, grad):
        """
        Propagate the gradient backward through every layer in reverse order.
        """
        for layer in reversed(self.layers):
            if layer.use_fused_gradient:
                grad = layer.backward_last_layer(grad)
            else:
                grad = layer.backward(grad)
                
                
    def update_weights(self):
        """Ask the optimizer to step each layer that has learnable parameters."""
        for layer in self.layers:
            self.optimizer.step(layer.get_parameters(), layer.get_gradients(), layer)


    def execute_training(self, X_train, y_train, X_val, y_val,
                         epochs=100, batch_size=32, early_stopping_patience=3):
        """
        Full mini-batch training loop.

        Each epoch:
          1. Shuffle training data (prevents the network memorising sample order)
          2. Iterate over mini-batches:
               a. Forward pass  → predictions ŷ
               b. Compute loss  L = criterion(ŷ, y)
               c. Backward pass → gradients dW, db
               d. Update weights  W ← W − lr·dW
          3. Evaluate on validation set (no weight update)
          4. Print metrics
          5. Early-stopping check: if val_loss has not improved → stop early to avoid overfitting
        """
        n         = X_train.shape[0]
        n_batches = (n + batch_size - 1) // batch_size

        train_loss_hist, val_loss_hist   = [], []
        train_acc_hist,  val_acc_hist    = [], []

        best_val_loss    = float("inf")
        patience_counter = 0
        epochs_run       = 0

        print(f"\n    Training samples : {n}")
        print(f"    Batch size       : {batch_size}")
        print(f"    Max epochs       : {epochs}")
        print(f"    Early stopping   : patience = {early_stopping_patience} epochs")
        print(f"\n    {'Epoch':>6}  {'Train Loss':>15}  {'Val Loss':>10}  "
              f"{'Train Acc':>10}  {'Val Acc':>8}")
        print(f"    {'─'*66}")

        for epoch in range(epochs):
            X_train, y_train = shuffle_data(X_train, y_train)
            total_loss = 0.0
            correct    = 0

            for start in range(0, n, batch_size):
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]

                preds       = self.forward(X_batch)
                total_loss += self.criterion.forward(preds, y_batch) * X_batch.shape[0]
                correct    += np.sum(np.argmax(preds,   axis=1) ==
                                     np.argmax(y_batch, axis=1))

                self.backward(self.criterion.backward())
                self.update_weights()

            val_loss,  val_acc  = self.evaluate(X_val,   y_val)
            train_loss = total_loss / n
            train_acc  = correct   / n

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            epochs_run += 1

            print(f"    {epoch+1:>6}/{epochs}  "
                  f"{train_loss:>11.4f}  {val_loss:>10.4f}  "
                  f"{train_acc:>10.4f}  {val_acc:>9.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\n    Early stopping triggered at epoch {epoch+1}.")
                print(f"    Best validation loss : {best_val_loss:.4f}")
                break

        print(f"    {'─'*66}")
        print(f"    Training complete — {epochs_run} epochs run.")

        show_combined_graph(epochs_run,
                            train_loss_hist, val_loss_hist,
                            train_acc_hist,  val_acc_hist)

        return {
            "train_loss": train_loss_hist,
            "val_loss":   val_loss_hist,
            "train_acc":  train_acc_hist,
            "val_acc":    val_acc_hist,
        }


    def fit_predict(self, X_val, y_val, bce_loss, batch_size=2, shuffle=False):
        """
        Inference loop: run the model on a dataset and display a confusion matrix.

        The fused model outputs a single raw logit per sample.
        We apply Sigmoid to turn it into a probability, then BCE to measure the loss.
        """
        n = X_val.shape[0]
        if shuffle:
            X_val, y_val = shuffle_data(X_val, y_val)

        total_loss = 0.0
        all_preds, all_true = [], []

        for start in range(0, n, batch_size):
            X_batch = X_val[start : start + batch_size]
            y_batch = y_val[start : start + batch_size]

            logits        = self.forward(X_batch)                        # raw output, shape (batch, 1)
            sigmoid_probs = 1 / (1 + np.exp(-logits))                    # Sigmoid → probability in (0, 1)
            bce           = bce_loss.forward(sigmoid_probs, y_batch)     # BCE loss for this batch
            total_loss   += bce * X_batch.shape[0]

            all_preds.extend((sigmoid_probs >= 0.5).astype(int).flatten().tolist())
            all_true.extend(y_batch.flatten().tolist())

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)
        loss      = total_loss / n
        accuracy  = np.mean(all_preds == all_true)

        print(f"\n    Inference results : {n} samples")
        print(f"    Loss     : {loss:.4f}")
        print(f"    Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")

        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(all_true.astype(int), all_preds.astype(int)):
            cm[t][p] += 1

        show_confusion_matrix(cm, loss, accuracy)


    def evaluate(self, X, y):
        """
        Compute loss and accuracy on a full dataset without updating weights.
        Called at the end of every training epoch for the validation set.
        """
        preds    = self.forward(X)
        loss     = self.criterion.forward(preds, y)
        accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy
