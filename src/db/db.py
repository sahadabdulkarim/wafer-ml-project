# src/db/db.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.sql import insert
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "wafer_data.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLITE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(SQLITE_URL, echo=False)
metadata = MetaData()

# simple tables for demo
wafers = Table(
    "wafers", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("wafer_name", String, nullable=False),
)

devices = Table(
    "devices", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("wafer_id", Integer),
    Column("device_name", String),
    Column("vth", Float),   # example metric
    Column("leakage", Float),
)

def init_db():
    metadata.create_all(engine)

def insert_device(wafer_name, device_name, vth, leakage):
    # ensure wafer exists
    with engine.begin() as conn:
        res = conn.execute(
            wafers.select().where(wafers.c.wafer_name == wafer_name)
        ).first()
        if res is None:
            r = conn.execute(insert(wafers).values(wafer_name=wafer_name))
            wafer_id = r.inserted_primary_key[0]
        else:
            wafer_id = res.id

        conn.execute(insert(devices).values(
            wafer_id=wafer_id,
            device_name=device_name,
            vth=vth,
            leakage=leakage
        ))
