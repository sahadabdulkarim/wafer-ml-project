# src/data/simulate.py
"""
Generate synthetic wafer device data with spatial layout and injected defects.
Appends into existing src/wafer_data.db devices + wafers tables.

Usage:
    python -m src.data.simulate
or
    python src\data\simulate.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

RNG = np.random.default_rng(42)

BASE_DIR = Path(__file__).resolve().parents[1]  # src/
DB_PATH = BASE_DIR / "wafer_data.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

def ensure_wafer(name):
    """Return wafer_id for wafer name, create if missing."""
    with ENGINE.begin() as conn:
        r = conn.execute(text("SELECT id FROM wafers WHERE wafer_name = :name"), {"name": name}).fetchone()
        if r is None:
            res = conn.execute(text("INSERT INTO wafers (wafer_name) VALUES (:name)"), {"name": name})
            # sqlite: lastrowid available via cursor, but SQLAlchemy 1.4+ returns nothing; re-query
            r = conn.execute(text("SELECT id FROM wafers WHERE wafer_name = :name"), {"name": name}).fetchone()
        return int(r[0])

def generate_wafer(wafer_name, n_devices=1000, grid=True, defect_clusters=2):
    """
    Generate a wafer dataset:
      - grid: create x,y grid positions (approx sqrt(n) x sqrt(n))
      - base Vth per wafer + gradual radial gradient
      - leakage ~ 10^( -9 + small noise ) correlated with Vth slightly
      - inject defect clusters with higher Vth or higher leakage
    Returns pandas.DataFrame with columns: wafer_id, device_name, x, y, vth, leakage, is_defect
    """
    wafer_id = ensure_wafer(wafer_name)

    # grid layout
    side = int(np.ceil(np.sqrt(n_devices)))
    xs = np.repeat(np.arange(side), side)[:n_devices]
    ys = np.tile(np.arange(side), side)[:n_devices]

    # Normalize coordinates to [-1,1] (center-based)
    xs_n = (xs - xs.mean()) / max(1, xs.max()-xs.min())
    ys_n = (ys - ys.mean()) / max(1, ys.max()-ys.min())

    # wafer-level baseline
    baseline_vth = 0.70 + RNG.normal(0, 0.01)   # around 0.7V
    # introduce wafer-wide radial gradient
    radius = np.sqrt(xs_n**2 + ys_n**2)
    gradient = -0.02 * radius  # edge slightly lower Vth

    # device Vth with local noise
    vth = baseline_vth + gradient + RNG.normal(0, 0.01, size=n_devices)

    # leakage: small values around 1e-9, correlated with vth negatively (higher vth -> lower leakage)
    leakage_base = 1e-9 * (1 + RNG.normal(0, 0.2, size=n_devices))
    leakage = leakage_base * (1.0 - 0.2 * (vth - baseline_vth))

    # prepare is_defect array (all zeros initially)
    is_defect = np.zeros(n_devices, dtype=int)

    # inject defects: some clusters get bumped leakage or vth
    defect_info = []
    for d in range(defect_clusters):
        cx = RNG.integers(0, side)
        cy = RNG.integers(0, side)
        radius_px = max(1, int(side * (0.05 + RNG.random()*0.07)))
        # compute distance in grid units
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        mask = dist <= radius_px
        if RNG.random() < 0.5:
            # high leakage region
            leakage[mask] *= 10 ** (1 + RNG.normal(0, 0.3))
            defect_info.append(("high_leak", cx, cy, radius_px, int(mask.sum())))
        else:
            # shifted Vth region
            vth[mask] += 0.06 + RNG.normal(0, 0.01)
            defect_info.append(("vth_shift", cx, cy, radius_px, int(mask.sum())))
        # mark ground-truth defects
        is_defect[mask] = 1

    # create names and DataFrame
    device_names = [f"DIE{str(i+1).zfill(4)}" for i in range(n_devices)]
    df = pd.DataFrame({
        "wafer_id": wafer_id,
        "device_name": device_names,
        "x": xs,
        "y": ys,
        "vth": vth,
        "leakage": np.abs(leakage),  # ensure positive
        "is_defect": is_defect
    })

    return df, defect_info

def append_to_db(df):
    # append to existing devices table
    df_to_write = df[["wafer_id","device_name","vth","leakage","x","y","is_defect"]].copy()
    df_to_write.to_sql("devices_sim", ENGINE, if_exists="append", index=False)
    print("Appended", len(df_to_write), "rows to devices_sim")


def main():
    # parameters: create 3 wafers with 1000 devices each
    wafers = ["W10", "W11", "W12"]
    all_defects = {}
    for w in wafers:
        df, defects = generate_wafer(w, n_devices=1000, defect_clusters=2)
        append_to_db(df)
        all_defects[w] = defects

    # show defect summary
    for w, info in all_defects.items():
        print(w, "defects:", info)

    print("Done. Check devices_sim table in DB (and later merge/rename as needed).")

if __name__ == "__main__":
    main()
