# AIS Vessel Trajectory Prediction

Predict future vessel positions (latitude/longitude) from AIS signals and maritime context.

**Highlights**
- Time-aware validation to avoid leakage
- **Two independent modeling approaches** (run separately or ensemble)

---

## Data

> The **training dataset is not included** (too large). Place it locally under `data/raw/train.csv`.

Included in this repo:
- `ais_test.csv`, `ais_sample_submission.csv`
- `merged_data.csv` (AIS + ports/schedules join)
- `ports.csv`, `vessels.csv`, `schedules_to_may_2024.csv`


## Notebooks & Models

You mentioned that the two short notebooks are **different models**. Here’s a clear mapping and suggested renames:

**Current → Suggested**
- `Report.ipynb` → `01_eda_and_feature_engineering.ipynb`  
  _EDA, data sanity checks, feature engineering, baselines._
- `Short_notebook1.ipynb` → `02_model_clustered_rf_by_port_distance.ipynb` (**Model A**)  
  _Clusters data by `distance_to_port`; trains RandomForest regressors per target/cluster._
- `Short_notebook2.ipynb` → `03_model_vessel_features_lgbm_rf.ipynb` (**Model B**)  
  _Vessel-aware features (rolling stats, metadata merges) with LightGBM/RandomForest._

### Model A — Clustered RF (distance-to-port)
- **File:** `notebooks/02_model_clustered_rf_by_port_distance.ipynb`
- **Idea:** Split tracks by `distance_to_port` bins/clusters; fit `RandomForestRegressor` per cluster/target.
- **Features:** temporal deltas, recent lags, speed/course dynamics, cluster tag.
- **Output:** `data/submissions/submission_modelA.csv`

### Model B — Vessel-aware GBM/RF
- **File:** `notebooks/03_model_vessel_features_lgbm_rf.ipynb`
- **Idea:** Merge AIS with `vessels.csv` (+ schedules) to build vessel-level rolling features; train LightGBM and/or RandomForest.
- **Features:** enriched vessel metadata, rolling/lagged signals, proximity features.
- **Output:** `data/submissions/submission_modelB.csv`


## Evaluation

- Validation uses **time-ordered** splits (e.g., last N days by vessel) to prevent leakage.
- Metrics: per-coordinate MAE and **Haversine** distance (meters).
- Each modeling notebook logs validation scores at the end.
