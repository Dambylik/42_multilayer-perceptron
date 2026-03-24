import argparse
import sys
import json
from tools.utils import split_dataset, section, subsection


def main():

    # ══════════════════════════════════════════════════════════════════════════
    # ARGUMENTS
    # ══════════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(
        description="Parse args and split dataset",
        usage="python3 main.py data.csv [options]"
    )
    parser.add_argument("dataset",         help="Raw dataset CSV (will be auto-split)")
    parser.add_argument("--epochs",        type=int,   default=70,   help="Number of epochs")
    parser.add_argument("--batch_size",    type=int,   default=16,   help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--layer",         type=int,   nargs="+",    help="Hidden layer widths")
    parser.add_argument("--split",         type=float, default=0.2,  help="Validation split ratio")
    parser.add_argument("--patience",      type=int,   default=3,    help="Early stopping patience")
    parser.add_argument("--adam",          action="store_true",       help="Use Adam instead of SGD")
    args = parser.parse_args()

    optimizer_name = "Adam" if args.adam else "SGD"
    arch_str       = str(args.layer) if args.layer else "[24, 10, 8]  (default)"

    section("ARGUMENTS")
    print(f"""
    All options you passed (or their defaults) are shown below.

        Dataset         : {args.dataset}
        Epochs          : {args.epochs}
        Batch size      : {args.batch_size}
        Learning rate   : {args.learning_rate}
        Optimizer       : {optimizer_name}
        Hidden layers   : {arch_str}
        Val split       : {int(args.split*100)} %
        Patience        : {args.patience}
    """)

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET SPLIT
    # ══════════════════════════════════════════════════════════════════════════
    section("DATASET SPLIT ")
    print(f"""
    The raw dataset is randomly divided into two NON-OVERLAPPING subsets:

    • Training set   → the network SEES these samples and adjusts weights
    • Validation set → the network NEVER trains on these; used only to
                       measure how well it generalises to UNSEEN data
    """)

    try:
        split_dataset(args.dataset, args.split)
    except (ValueError, FileNotFoundError) as err:
        print(f"  Error: {err}")
        sys.exit(1)

    # ── Save session.json ─────────────────────────────────────────────────────
    session = {
        "dataset":       args.dataset,
        "train_path":    "generated/train_set.csv",
        "val_path":      "generated/validation_set.csv",
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.learning_rate,
        "layer":         args.layer,          # None = use default architecture
        "split":         args.split,
        "patience":      args.patience,
        "adam":          args.adam,
    }

    with open("generated/session.json", "w") as f:
        json.dump(session, f, indent=4)



if __name__ == "__main__":
    main()
