# Batch Prediction Pipeline

A simulated real-world ML batch prediction system using SQLite, scikit-learn, and APScheduler.

## Project Structure

```
batch-prediction-pipeline/
├── README.md
├── requirements.txt
├── train_model.py       # Train and save the ML model
├── batch_predict.py     # Core batch prediction script
├── scheduler.py         # Run batch_predict.py on a schedule
└── db/
    └── init_db.py       # Initialize the database and seed input data
```

## How It Works

1. **Database** — SQLite with two tables:
   - `input_data` — raw feature rows waiting to be predicted
   - `predictions` — results written back after each batch run

2. **Model** — RandomForestClassifier trained on the Iris dataset, saved as `model.pkl`

3. **Batch script** — reads unpredicted rows, loads the model, writes predictions back

4. **Scheduler** — runs the batch script automatically every 5 minutes

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize DB and seed input data
python db/init_db.py

# 3. Train and save the model
python train_model.py

# 4a. Run a single batch prediction manually
python batch_predict.py

# 4b. OR start the scheduler (runs every 5 minutes automatically)
python scheduler.py
```

## Database Schema

```sql
CREATE TABLE input_data (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sepal_length REAL NOT NULL,
    sepal_width  REAL NOT NULL,
    petal_length REAL NOT NULL,
    petal_width  REAL NOT NULL
);

CREATE TABLE predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id             INTEGER NOT NULL,
    prediction           TEXT NOT NULL,
    prediction_timestamp TEXT NOT NULL,
    FOREIGN KEY (input_id) REFERENCES input_data(id)
);
```

## Requirements

- Python 3.8+
- scikit-learn
- APScheduler
- joblib
- pandas
