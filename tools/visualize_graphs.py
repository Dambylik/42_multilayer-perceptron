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


def show_confusion_matrix(cm, loss, accuracy):
    """Print confusion matrix stats and save the plot."""
    import numpy as np

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    print(f"\n    Confusion matrix:")
    print(f"                    Predicted B              Predicted M")
    print(f"      Actual B   :  TN = {tn:4d}  (correct)   FP = {fp:4d}  ← false alarm")
    print(f"      Actual M   :  FN = {fn:4d}  ← missed cancer (DANGEROUS)   TP = {tp:4d}  (correct)")

    precision = tp / (tp + fp)          if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)          if (tp + fn) > 0 else 0.0
    f1        = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    print(f"\n    Precision : {precision:.4f}  (of predicted M, how many were actually M)")
    print(f"    Recall    : {recall:.4f}  (of actual M, how many did we catch)  ← most critical")
    print(f"    F1 score  : {f1:.4f}  (mean of Precision and Recall : penalises if either is low)")

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
                    ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=150)
    print("\n    Saved images/confusion_matrix.png")


def show_multi_run_graph(all_runs):
    """
    Overlay loss and accuracy curves from multiple training runs on the same graph.
    Each run must be a dict with keys: label, train_loss, val_loss, train_acc, val_acc.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-run comparison", fontsize=14)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, run in enumerate(all_runs):
        color  = colors[i % len(colors)]
        label  = run["label"]
        epochs = range(len(run["train_loss"]))

        ax_loss.plot(epochs, run["train_loss"], color=color, label=f"{label} — train")
        ax_loss.plot(epochs, run["val_loss"],   color=color, label=f"{label} — val", linestyle="--")

        ax_acc.plot(epochs, run["train_acc"], color=color, label=f"{label} — train")
        ax_acc.plot(epochs, run["val_acc"],   color=color, label=f"{label} — val", linestyle="--")

    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss — all runs")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy — all runs")
    ax_acc.legend(fontsize=7)
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    out = "images/multi_run_curves.png"
    plt.savefig(out, dpi=150)
    print(f"    Saved {out}")


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
