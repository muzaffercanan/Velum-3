# CS440/540 HW3 Solutions

This repository contains the completed solutions for HW3 (ML for finance / time series) across four problems. All code is implemented in `CS440-540_HW3.ipynb` and supports local execution with PyTorch and TA-Lib.

## Contents
- `CS440-540_HW3.ipynb` ? notebook with solutions for Problems 1?4
- `AAPL.csv` ? Apple price/volume data
- `fundamentals.csv` ? company fundamentals data for EPS prediction
- `household_power_consumption.txt` ? household energy consumption data

## Problem Summaries
- **Problem 1 (AAPL MLP)**: Predict close price using previous 5 days of close/volume. Evaluated 4 MLP configs; best MAPE ? 1.9%, RMSE ? $5.3.
- **Problem 2 (BTC CNN with TA-Lib)**: Compute MACD/RSI/CMO/MOM/BBands/SMA via TA-Lib, build 6?6 images from 6-day windows, 2D CNN for next-day up/down. Best run: Accuracy ? 0.49, F1 ? 0.46 (baseline-level).
- **Problem 3 (EPS LSTM)**: Multivariate LSTM over 4-quarter sequences of fundamentals to predict EPS. Hyperparameter search over hidden sizes [5, 10, 30]; best hidden=10, Test MAPE ? 0.50.
- **Problem 4 (CNN-LSTM power)**: Forecast 60 minutes ahead from 600-minute history of `Global_active_power`. CNN?LSTM?FC; stride/windowing to control runtime. Test RMSE ? 0.92 kW on a 150k-row subset.

## Environment
- Python 3.12
- PyTorch (CPU) installed via pip (user site)
- TA-Lib installed via pip (user site)
- yfinance for BTC data download

If TA-Lib is not found in your kernel, run inside the notebook:
```
!python -m pip install --user TA-Lib
```
Then restart the kernel and re-run cells.

## Running
1. Ensure dependencies: `pip install --user torch torchvision torchaudio ta-lib yfinance scikit-learn`
2. Open `CS440-540_HW3.ipynb` in Jupyter/VS Code and run the solution cells.
3. For Problem 4, adjust `max_rows`, `stride`, and `epochs` in the cell to balance runtime vs. accuracy; set `max_rows=None` to use all data.

## Notes
- All splits are chronological to avoid leakage.
- Features/targets are scaled using train statistics where appropriate.
- Seeds are fixed for basic reproducibility.
