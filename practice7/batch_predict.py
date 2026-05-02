"""
batch_predict.py
----------------
Core batch prediction script.

Steps:
  1. Connect to the database
  2. Read input rows that have not yet been predicted
  3. Load the trained model from disk
  4. Generate predictions
  5. Write results back to the predictions table

Usage:
    python batch_predict.py
"""

import logging
import sqlite3
from datetime import datetime

import joblib
import pandas as pd

DB_PATH = "pipeline.db"
MODEL_PATH = "model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_predict")

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def fetch_unpredicted(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return rows from input_data that have no matching prediction yet."""
    query = """
        SELECT id, sepal_length, sepal_width, petal_length, petal_width
        FROM   input_data
        WHERE  id NOT IN (SELECT input_id FROM predictions)
    """
    return pd.read_sql_query(query, conn)


def save_predictions(
    conn: sqlite3.Connection,
    input_ids: list,
    labels: list,
    timestamp: str,
) -> None:
    rows = [(int(iid), lbl, timestamp) for iid, lbl in zip(input_ids, labels)]
    conn.executemany(
        "INSERT INTO predictions (input_id, prediction, prediction_timestamp)"
        " VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()


def run() -> None:
    log.info("=== Batch prediction started ===")

    # 1. Connect
    conn = sqlite3.connect(DB_PATH)

    # 2. Read unpredicted input data
    df = fetch_unpredicted(conn)

    if df.empty:
        log.info("No new rows found — nothing to predict.")
        conn.close()
        return

    log.info("Found %d rows to predict", len(df))

    # 3. Load model
    model = joblib.load(MODEL_PATH)

    # 4. Generate predictions
    X = df[FEATURE_COLS].values
    pred_indices = model.predict(X)
    labels = [model.class_names_[i] for i in pred_indices]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 5. Write results
    save_predictions(conn, df["id"].tolist(), labels, timestamp)
    log.info("Saved %d predictions at %s", len(labels), timestamp)

    conn.close()
    log.info("=== Batch prediction finished ===")


if __name__ == "__main__":
    run()
