"""
STEP 4 — PREDICT
"""

import argparse
from src.loss_functions import BinaryCrossEntropy
from tools.utils import (section, reconstruct_model_from_json,
                         fuse_softmax_to_sigmoid, build_inference_model,
                         load_dataset)


def run_inference():
    cli_parser = argparse.ArgumentParser(description="MLP Prediction Utility")
    cli_parser.add_argument("data_file",  nargs="?", default="generated/validation_set.csv")
    cli_parser.add_argument("model_file", nargs="?", default="generated/export.json")
    args = cli_parser.parse_args()

    try:
        layers = reconstruct_model_from_json(args.model_file)
    except Exception as error:
        print(f"  Error loading model: {error}")
        return

    try:
        val_features, val_labels, _, _ = load_dataset(args.data_file, one_hot=False)
    except Exception as error:
        print(f"  Error loading data: {error}")
        return

    layers = fuse_softmax_to_sigmoid(layers)
    model  = build_inference_model(layers)

    section("RUN INFERENCE")
    model.fit_predict(val_features, val_labels, BinaryCrossEntropy(), batch_size=4, shuffle=False)


if __name__ == "__main__":
    run_inference()
