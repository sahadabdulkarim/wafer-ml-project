# drop_sim_tables.py
from sqlalchemy import create_engine, text
from pathlib import Path

engine = create_engine(f"sqlite:///{Path('src/wafer_data.db').resolve().as_posix()}")

with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS devices_sim"))
    conn.execute(text("DROP TABLE IF EXISTS devices_sim_ml"))
print("Dropped devices_sim and devices_sim_ml (if existed)")
