# Week 4: Unsupervised Learning, Time Series, and Reporting

This week covers clustering, PCA, forecasting, and reproducible reporting for the final project.

## Lecture_Support Notebooks
- `01_Week4_Day1_Clustering_PCA.ipynb`
- `02_Week4_Day2_Time_Series_Forecasting.ipynb`
- `03_Week4_Day3_Reporting_Reproducibility_Final_Project.ipynb`

## Shared data (`data/week4/`)
- `week4_customer_segmentation.csv` (created by Day 1)
- `week4_air_passengers.csv` (created by Day 2)
- `week4_forecast_model_comparison.csv` (created by Day 2)
- `week4_environment_versions.csv` (created by Day 3)

## Figures (`figures/`)
Saved plots from the notebooks (feature distributions, PCA, forecasts, etc.).

## Suggested run order
1. Day 1 — Clustering + PCA
2. Day 2 — Time series forecasting (set `RUN_DEEP_LEARNING = False` if training is slow)
3. Day 3 — Reporting, reproducibility, final project template

## Optional packages
- `prophet` — Prophet forecasts (Day 2)
- `xgboost` — gradient boosting (Day 2 falls back to HistGradientBoosting)
- `torch` — LSTM, GRU, Transformer demos (Day 2)

Install extras: `pip install prophet xgboost torch`

## Regenerate notebooks
```bash
python scripts/build_week4_notebooks.py
```
