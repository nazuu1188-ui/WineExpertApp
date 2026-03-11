import sqlite3
import json
from typing import Optional
import pandas as pd

def init_db(db_path: str = "runs.sqlite") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT,
        dataset_name TEXT,
        dataset_hash TEXT,
        n_objects INTEGER,
        n_attributes INTEGER,
        params_json TEXT,
        runtime_sec REAL,
        status TEXT,
        error_msg TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS exports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        file_type TEXT,
        file_path TEXT
    )
    """)

    # Store per-object COCO output (so DB is "mineable")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coco_std (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        object_id TEXT,
        estimation REAL,
        actual REAL,
        delta REAL,
        delta_pct REAL,
        validation INTEGER,
        conclusion TEXT
    )
    """)

    conn.commit()
    return conn

def log_run_start(conn, run_id: str, created_at: str, dataset_name: str, dataset_hash: str,
                  n_objects: int, n_attributes: int, params: dict):
    conn.execute("""
    INSERT INTO runs(run_id, created_at, dataset_name, dataset_hash, n_objects, n_attributes, params_json, runtime_sec, status, error_msg)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (run_id, created_at, dataset_name, dataset_hash, n_objects, n_attributes, json.dumps(params), None, "running", None))
    conn.commit()

def log_run_end(conn, run_id: str, runtime_sec: float, status: str, error_msg: Optional[str] = None):
    conn.execute("""
    UPDATE runs SET runtime_sec=?, status=?, error_msg=? WHERE run_id=?
    """, (runtime_sec, status, error_msg, run_id))
    conn.commit()

def log_export(conn, run_id: str, file_type: str, file_path: str):
    conn.execute("""
    INSERT INTO exports(run_id, file_type, file_path) VALUES(?,?,?)
    """, (run_id, file_type, file_path))
    conn.commit()

def store_coco_std(conn, run_id: str, conclusion_df: pd.DataFrame):
    """
    Expects columns: OAM, Becslés, Tény+0, Delta, Delta/Tény, validation, conclusion
    """
    df = conclusion_df.copy()
    needed = ["OAM", "Becslés", "Tény+0", "Delta", "Delta/Tény", "validation", "conclusion"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"store_coco_std missing column: {c}")

    rows = []
    for _, r in df.iterrows():
        rows.append((
            run_id,
            str(r["OAM"]),
            float(r["Becslés"]) if r["Becslés"] is not None else None,
            float(r["Tény+0"]) if r["Tény+0"] is not None else None,
            float(r["Delta"]) if r["Delta"] is not None else None,
            float(r["Delta/Tény"]) if r["Delta/Tény"] is not None else None,
            int(r["validation"]) if r["validation"] is not None else None,
            str(r["conclusion"]) if r["conclusion"] is not None else None,
        ))

    conn.executemany("""
    INSERT INTO coco_std(run_id, object_id, estimation, actual, delta, delta_pct, validation, conclusion)
    VALUES(?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()