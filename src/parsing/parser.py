# src/parsing/parser.py
import json
import pandas as pd
from pathlib import Path
from src.db.db import init_db, insert_device


DATA_DIR = Path(__file__).resolve().parents[2] / "data_raw"

def parse_csv(file_path):
    # assume CSV has columns: wafer_name, device_name, vth, leakage
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        insert_device(row['wafer_name'], row['device_name'],
                      float(row['vth']), float(row['leakage']))

def parse_json(file_path):
    with open(file_path, 'r') as f:
        payload = json.load(f)
    # support two shapes: list of records OR dict with 'measurements'
    records = []
    if isinstance(payload, dict) and "measurements" in payload:
        records = payload["measurements"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Unexpected JSON shape")

    for rec in records:
        # try to extract fields robustly
        wafer = rec.get("wafer_name") or rec.get("wafer")
        device = rec.get("device_name") or rec.get("device")
        vth = rec.get("vth") or rec.get("Vth") or rec.get("threshold_voltage")
        leakage = rec.get("leakage") or rec.get("I_leak") or rec.get("leakage_current")
        insert_device(wafer, device, float(vth), float(leakage))

def run_demo():
    init_db()
    for f in DATA_DIR.glob("*"):
        if f.suffix.lower() == ".csv":
            parse_csv(f)
        elif f.suffix.lower() == ".json":
            parse_json(f)

if __name__ == "__main__":
    run_demo()
    print("Done parsing.")
