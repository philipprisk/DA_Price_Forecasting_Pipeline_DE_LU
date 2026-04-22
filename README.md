# DA Price Forecasting Pipeline DE-LU

Code for day-ahead electricity price forecasting for the DE-LU bidding zone using LEAR (Least Absolute Shrinkage and Selection Operator with Autoregressive terms) and SQRA (Sequential Quantile Regression Averaging).

## Repository Structure
DA_Price_Forecasting_Pipeline_DE_LU/
├── data/                  # Clustering files and spatial data
├── evaluation/            # Evaluation notebook
├── pipeline/
│   ├── lear/              # LEAR point forecast and ANC analysis
│   └── sqra/              # SQRA quantile forecast
├── preprocessing/         # ERA5 and ICON-D2 aggregation scripts
├── requirements/          # Per-module requirements files
└── visualization/         # Plotting scripts

## Setup

Each module has its own virtual environment and requirements file in `requirements/`.

## Workflow

1. Run `pipeline/lear/lear_pipeline.ipynb` to generate point forecasts
2. Run `pipeline/sqra/sqra_pipeline.ipynb` to generate quantile forecasts
3. Run `evaluation/evaluation.ipynb` for model evaluation
