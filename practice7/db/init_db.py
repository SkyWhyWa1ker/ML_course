"""
db/init_db.py
-------------
Creates the SQLite database, initialises the schema,
and seeds the input_data table with sample Iris rows.

Usage:
    python db/init_db.py
"""

import sqlite3
import numpy as np
from sklearn.datasets import load_iris

DB_PATH = "pipeline.db"


def create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            sepal_length REAL NOT NULL,
            sepal_width  REAL NOT NULL,
            petal_length REAL NOT NULL,
            petal_width  REAL NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id             INTEGER NOT NULL,
            prediction           TEXT    NOT NULL,
            prediction_timestamp TEXT    NOT NULL,
            FOREIGN KEY (input_id) REFERENCES input_data(id)
        )
    """)

    conn.commit()
    print(f"[init_db] Tables created in '{DB_PATH}'")


def seed_input_data(conn: sqlite3.Connection, n: int = 50) -> None:
    iris = load_iris()
    idx = np.random.choice(len(iris.data), size=n, replace=False)
    rows = [
        (
            float(iris.data[i][0]),
            float(iris.data[i][1]),
            float(iris.data[i][2]),
            float(iris.data[i][3]),
        )
        for i in idx
    ]

    conn.executemany(
        "INSERT INTO input_data (sepal_length, sepal_width, petal_length, petal_width)"
        " VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"[init_db] Seeded {n} rows into input_data")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    seed_input_data(conn, n=50)
    conn.close()
    print("[init_db] Done.")


if __name__ == "__main__":
    main()
