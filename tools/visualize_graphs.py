import matplotlib.pyplot as plt
import numpy as np


def show_combined_graph(epochs, loss_history, loss_history_validation, accuracy_history, accuracy_history_validation):
    """Display loss and accuracy learning curves side-by-side on one figure."""
    try:
        epochs_graph = range(0, epochs)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Training metrics", fontsize=14)
        ax_loss.plot(epochs_graph, loss_history, label="training loss", color='royalblue')
        ax_loss.plot(epochs_graph, loss_history_validation, label="validation loss", color='tomato', linestyle='--')
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.set_ylim(0, max(np.max(loss_history), np.max(loss_history_validation)))
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        ax_acc.plot(epochs_graph, accuracy_history, label="training accuracy",  color='royalblue')
        ax_acc.plot(epochs_graph, accuracy_history_validation, label="validation accuracy", color='tomato', linestyle='--')
        ax_acc.set_xlabel("Epochs")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.set_ylim(
            min(np.min(accuracy_history), np.min(accuracy_history_validation)),
            max(np.max(accuracy_history), np.max(accuracy_history_validation)),
        )
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("images/training_curves.png", dpi=150)
        print("    Saved images/training_curves.png")        
    except KeyboardInterrupt:
        print("Ctrl+C: closing plot window and exiting program")
        plt.close("all")


def show_graph_loss(epochs, loss_history, loss_history_validation):
    try:
        epochs_graph = range(0, epochs)
        train_loss = loss_history
        val_loss = loss_history_validation
        plt.figure()
        plt.plot(epochs_graph, train_loss, label="training loss")
        plt.plot(epochs_graph, val_loss, label="validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss train vs loss validation")
        train_loss_max = np.max(train_loss)
        val_loss_max = np.max(val_loss)
        plt.ylim(0, max(train_loss_max, val_loss_max))
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        print("Ctrl+C: closing plot window and exiting program")
        plt.close('all')



def show_graph_accuracy(epochs, accuracy_history, accuracy_history_validation) :
    try :
        epochs_graph = range(0, epochs) 
        train_accuracy= accuracy_history
        val_accuracy = accuracy_history_validation
        plt.figure()
        plt.plot(epochs_graph, train_accuracy, label="training accuracy")
        plt.plot(epochs_graph, val_accuracy, label="validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy train vs accuracy validation")
        train_accuracy_max = np.max(train_accuracy)
        val_accuracy_max = np.max(val_accuracy)
        train_accuracy_min = np.min(train_accuracy)
        val_accuracy_min = np.min(val_accuracy)
        plt.ylim(min(train_accuracy_min, val_accuracy_min), max(train_accuracy_max, val_accuracy_max))
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        print("Ctrl+C: closing plot window and exiting program")
        plt.close('all')
