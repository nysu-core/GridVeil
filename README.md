# GridVeil

GridVeil is a CNN-LSTM-attention forecasting model for three power-consumption zones.

## Reproducibility status

- The project uses a chronological 70/15/15 train/validation/test split in `data_preprocessing.py`.
- `train.py` reports MAPE and anomaly errors after inverse-scaling to the original MW units.
- The anomaly threshold is the 95th percentile of training absolute errors, also in MW. The script prints the resulting value; no fixed value such as `0.421` should be cited without a fresh run.
- The old `7,834` anomaly figure is not supported by this repository. It was a window/row count, not a verified anomaly count.
- The old `5.77%` Random Forest figure is not adopted as a result. Run `random_forest_baseline.py` to generate a baseline under the same split.
- `dp_experiment.py` provides a reproducible Laplace label-perturbation experiment. It is not a formal end-to-end differential-privacy implementation and must not be reported as one without a privacy accountant and training mechanism.
- The private dataset is intentionally ignored by Git via `.gitignore`.

Install the dependencies from `requirements.txt`, then run the scripts from the repository root.
