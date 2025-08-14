# AIS Vessel Trajectory Prediction

Predict future vessel positions (latitude/longitude) from AIS signals and maritime context. Random Forest and LightGBM.

---

## Data

The models are mainly trained on publically available AIS data.

- ais_train.csv: Training data. This contains the positions of
689 vessels. The dataset was sampled every 20 minutes, but the timestamps
for each vessel are irregular.
- schedules_to_may_2024.csv: Contains the planned arrival destinations and
time as communicated from the shipping lines for a select 252 vessels.
- vessels.csv: Data about each vessel
- ports.csv: Contains information about the ports referenced in



## Notebooks & Models


- `01_eda_and_feature_engineering.ipynb`  
  _EDA, data sanity checks, feature engineering, baselines._
- `02_model_clustered_rf_by_port_distance.ipynb` (**Model A**)  
  _Clusters data by `distance_to_port`; trains RandomForest regressors per target/cluster._
- `03_model_vessel_features_lgbm_rf.ipynb` (**Model B**)  
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
