# src/ml/pipeline.py
"""
Simple ML pipeline:
- loads devices from src/wafer_data.db
- scales numeric features
- runs KMeans and IsolationForest
- writes a devices_ml table back to the same DB with results
Run as:
    python -m src.ml.pipeline
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Config
BASE_DIR = Path(__file__).resolve().parents[1]   # src/
DB_PATH = BASE_DIR / "wafer_data.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
RANDOM_STATE = 42

def load_devices():
    df = pd.read_sql("SELECT * FROM devices", ENGINE)
    return df

def preprocess(df):
    # make numeric and pick features vth/leakage if available
    features = []
    for c in ['vth', 'leakage', 'x', 'y']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows missing core metrics
    # require vth and leakage and coords (if coords exist in table)
    req = ['vth', 'leakage']
    df = df.dropna(subset=req).reset_index(drop=True)
    # keep only positive leakage
    df = df[df['leakage'] > 0].reset_index(drop=True)

    # log-transform leakage
    df['log_leakage'] = np.log10(df['leakage'])
    features = ['vth', 'log_leakage']

    # add radial distance if coordinates exist
    if {'x', 'y'}.issubset(df.columns):
        # center coordinates and normalize to unit scale
        x_mean, y_mean = df['x'].mean(), df['y'].mean()
        df['radial_dist'] = np.sqrt((df['x'] - x_mean)**2 + (df['y'] - y_mean)**2)
        # normalize radial dist to zero mean, unit std (scaler will do this later, but keep numeric)
        features.append('radial_dist')

    return df, features

def run_kmeans(X, k=3):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    return labels, km

def run_isolation_forest(X, contamination=0.01):
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=RANDOM_STATE)
    iso.fit(X)
    scores = iso.decision_function(X)       # higher -> more normal
    preds = iso.predict(X)                  # -1 anomaly, 1 normal
    is_anomaly = preds == -1
    return is_anomaly, scores, iso

def save_results(results_df):
    # results_df should have 'id', 'cluster_label', 'is_anomaly', 'anomaly_score'
    # write to DB table devices_ml (replace)
    results_df.to_sql("devices_ml", ENGINE, if_exists="replace", index=False)
    print("Wrote devices_ml table (rows):", len(results_df))


def main(k=3, contamination=0.01):
    df = load_devices()
    print("Loaded devices rows:", len(df))
    df_proc, features = preprocess(df)
    print("After preprocessing:", len(df_proc), "features used:", features)
    if len(df_proc) == 0:
        print("No usable rows after preprocessing. Exiting.")
        return

    X_raw = df_proc[features].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # K-Means
    labels, kmodel = run_kmeans(X, k=k)
    print("KMeans complete. Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index())

    # Isolation Forest
    contamination_val = 0.03   
    is_anom, scores, iso_model = run_isolation_forest(X, contamination=contamination_val)
    print(f"IsolationForest (contamination={contamination_val}): anomalies:", is_anom.sum(), "out of", len(is_anom))


    results = pd.DataFrame({
        'id': df_proc['id'],
        'cluster_label': labels,
        'is_anomaly': is_anom.astype(int),
        'anomaly_score': scores
    })

    # optionally merge extra columns for convenience
    combined = df_proc.merge(results, on='id')
    save_results(combined[['id','cluster_label','is_anomaly','anomaly_score']])

    # quick console summary
    print("Summary per cluster (anomaly counts):")
    print(combined.groupby('cluster_label')['is_anomaly'].sum())
    # save a CSV copy for quick inspection
    combined.to_csv(BASE_DIR / "ml_results.csv", index=False)
    print("Also wrote ml_results.csv to src/")

if __name__ == "__main__":
    main(k=3, contamination=0.01)
