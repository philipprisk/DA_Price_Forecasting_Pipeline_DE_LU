# Day-Ahead Price Forecasting Pipeline DE-LU
This repository contains the complete forecasting pipeline accompanying the bachelor thesis *"An Open-Source Probabilistic Forecasting Pipeline for German Day-Ahead Prices"* at the Karlsruhe Institute of Technology (KIT).
The pipeline generates probabilistic day-ahead electricity price forecasts for the EPEX DE-LU bidding zone at 15-minute resolution (96 MTUs per day), combining LEAR point forecasting with SQRA probabilistic post-processing. All input data are sourced exclusively from open-access, automatically retrievable data sources available before the EPEX DE-LU auction closure at 12:00 CET.
> **Note**: The current implementation is research-oriented rather than fully operational. Individual pipeline stages require manual execution. Full automation represents the most immediate direction for future work, as discussed in the thesis.
## Forecast Configurations
Three configurations are supported, reflecting different information sets and publication times:
- **Fundamental Model**: Uses weather (DWD ICON-D2 or ERA5), load forecasts, lagged EPEX DE-LU prices, and calendar features. Available from approximately 10:30 CET.
- **EXAA-Enriched Model**: Augments the Fundamental Model with intra-zonal EXAA day-ahead prices. Available from approximately 11:15 CET.
- **EXAA-Only Model**: Uses exclusively EXAA prices as input features while retaining all pipeline transformations. Recommended configuration once EXAA prices are available.
## Repository Structure
```text
DA_Price_Forecasting_Pipeline_DE_LU/
├── data/                  # Clustering files and spatial data (shapefiles)
├── evaluation/            # Model evaluation notebook
├── pipeline/
│   ├── lear/              # LEAR point forecast and ANC feature importance
│   └── sqra/              # SQRA quantile forecast
├── preprocessing/         # ERA5 and ICON-D2 aggregation scripts
├── requirements/          # Per-module requirements files
├── results/               # Forecast outputs and evaluation metrics
└── visualization/         # Plotting scripts
```
## Data
The pipeline requires the following external data sources:
- **ENTSO-E Transparency Platform**: EPEX DE-LU day-ahead prices, EXAA prices, and load forecasts. Access via API key (set `ENTSOE_API_KEY` in `.env`, see `.env.example`).
- **DWD ICON-D2** (primary weather source): Operational NWP forecasts retrieved from the DWD Open Data Server. Pre-processed with `preprocessing/load_aggregate_save_ICON_GITHUB.py`. Set path in the Settings cell of `lear_pipeline.ipynb`.
- **ERA5 Reanalysis** (benchmark weather source): Retrieved from the Copernicus Climate Data Store (CDS). Pre-processed with `preprocessing/aggregate_save_ERA5_GITHUB.py`. Set paths in the Settings cell of `lear_pipeline.ipynb`.
## Setup
Each pipeline module has its own virtual environment:
```bash
python3 -m venv pipeline/lear/.venv
pipeline/lear/.venv/bin/pip install -r requirements/lear_requirements.txt

python3 -m venv pipeline/sqra/.venv
pipeline/sqra/.venv/bin/pip install -r requirements/sqra_requirements.txt

# Register the LEAR kernel for the evaluation notebook
pipeline/lear/.venv/bin/python -m ipykernel install --user --name lear_venv --display-name "Python (lear)"
```
Copy `.env.example` to `.env` and fill in your ENTSO-E API key:
```bash
cp .env.example .env
```
## Workflow
1. Run `pipeline/lear/lear_pipeline.ipynb` to generate point forecasts and ANC feature importance
2. Run `pipeline/sqra/sqra_pipeline.ipynb` to generate quantile forecasts
3. Run `evaluation/evaluation.ipynb` for model evaluation
