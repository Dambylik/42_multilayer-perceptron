"""
STEP 4 — PREDICT

Usage:
    python3 4_predict.py                          # uses generated/export.json
    python3 4_predict.py generated/export_run2.json
"""

import sys
from src.loss_functions import BinaryCrossEntropy
from tools.utils import (section, reconstruct_model_from_json,
                         fuse_softmax_to_sigmoid, build_inference_model,
                         load_dataset)


def run_inference():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "generated/export.json"

    try:
        layers = reconstruct_model_from_json(model_path)
    except Exception as error:
        print(f"  Error loading model: {error}")
        return

    try:
        val_features, val_labels, _, _ = load_dataset("generated/validation_set.csv", one_hot=False)
    except Exception as error:
        print(f"  Error loading data: {error}")
        return

    layers = fuse_softmax_to_sigmoid(layers)
    model  = build_inference_model(layers)

    section("RUN INFERENCE")
    model.fit_predict(val_features, val_labels, BinaryCrossEntropy(), batch_size=4, shuffle=False)


if __name__ == "__main__":
    run_inference()
