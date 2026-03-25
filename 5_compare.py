"""
STEP 5 — COMPARE MULTIPLE TRAINING RUNS
Plots all runs saved in generated/histories.json on the same graph.
"""

import json
from tools.utils import section
from tools.visualize_graphs import show_multi_run_graph


def main():
    histories_path = "generated/histories.json"
    try:
        with open(histories_path) as f:
            all_runs = json.load(f)
    except FileNotFoundError:
        print(f"  Error: {histories_path} not found. Run 3_train.py at least once.")
        return
    except json.JSONDecodeError:
        print(f"  Error: {histories_path} is corrupted.")
        return

    if not all_runs:
        print("  No runs found in histories.json.")
        return

    section("MULTI-RUN COMPARISON")
    print(f"\n    {len(all_runs)} run(s) found:\n")
    for i, run in enumerate(all_runs):
        epochs = len(run["train_loss"])
        best_val = min(run["val_loss"])
        print(f"      {i+1}. {run['label']}")
        print(f"         epochs run: {epochs}  |  best val_loss: {best_val:.4f}")

    show_multi_run_graph(all_runs)


if __name__ == "__main__":
    main()
