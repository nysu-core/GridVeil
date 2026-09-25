"""Reproducible Laplace label-perturbation experiment.

This is a bounded-label privacy experiment, not an end-to-end differential
privacy guarantee for the trained neural network. Report it separately from
formal DP results unless a DP accountant and training mechanism are added.
"""

import argparse

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from data_preprocessing import preprocess_data
from model import build_hybrid_model
from train import final_timestep_targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epsilon <= 0 or args.clip <= 0:
        raise ValueError("epsilon and clip must be positive")

    rng = np.random.default_rng(args.seed)
    processed_data = preprocess_data()
    train_features, train_target_windows = processed_data["sequences"]["train"]
    test_features, test_target_windows = processed_data["sequences"]["test"]
    target_scaler = processed_data["target_scaler"]
    train_targets = final_timestep_targets(train_target_windows)
    test_targets = final_timestep_targets(test_target_windows)

    # Clip standardized labels before adding Laplace noise. This bounds the
    # label contribution; it does not provide a full training-process DP proof.
    clipped_targets = np.clip(train_targets, -args.clip, args.clip)
    noise_scale = 2 * args.clip / args.epsilon
    noisy_targets = clipped_targets + rng.laplace(
        loc=0.0, scale=noise_scale, size=clipped_targets.shape
    )

    model = build_hybrid_model(train_features.shape[1:])
    model.fit(
        train_features,
        noisy_targets,
        epochs=args.epochs,
        batch_size=64,
        validation_split=0.0,
        verbose=1,
    )
    predictions = model.predict(test_features, batch_size=64, verbose=1)
    actual = target_scaler.inverse_transform(test_targets)
    predicted = target_scaler.inverse_transform(predictions)
    print(f"Label-perturbation MAPE: {mean_absolute_percentage_error(actual, predicted):.6%}")
    print(f"epsilon={args.epsilon}, clip={args.clip}, noise_scale={noise_scale}")


if __name__ == "__main__":
    main()
