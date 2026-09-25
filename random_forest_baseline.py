"""Reproducible Random Forest baseline for GridVeil forecasting."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from data_preprocessing import preprocess_data


RANDOM_STATE = 42
N_ESTIMATORS = 200


def flatten_windows(windows):
    """Convert each temporal feature window into one tabular feature vector."""
    return windows.reshape(windows.shape[0], -1)


def main():
    """Train and evaluate the baseline using the project's chronological split."""
    processed_data = preprocess_data()
    train_features, train_target_windows = processed_data["sequences"]["train"]
    test_features, test_target_windows = processed_data["sequences"]["test"]
    target_scaler = processed_data["target_scaler"]

    train_targets = train_target_windows[:, -1, :]
    test_targets = test_target_windows[:, -1, :]
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(flatten_windows(train_features), train_targets)
    predictions = model.predict(flatten_windows(test_features))

    actual = target_scaler.inverse_transform(test_targets)
    predicted = target_scaler.inverse_transform(predictions)
    print(f"RF MAE (MW): {mean_absolute_error(actual, predicted):.6f}")
    print(f"RF RMSE (MW): {np.sqrt(mean_squared_error(actual, predicted)):.6f}")
    print(f"RF MAPE: {mean_absolute_percentage_error(actual, predicted):.6%}")
    print(f"RF R-squared: {r2_score(actual, predicted):.6f}")


if __name__ == "__main__":
    main()
