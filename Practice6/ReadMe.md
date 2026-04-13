
This project demonstrates how to train a simple machine learning model and deploy it as an API using FastAPI and Docker.

## Project Structure

- `train.py` — trains and saves the model
- `main.py` — FastAPI app
- `model.joblib` — saved model
- `requirements.txt` — dependencies
- `Dockerfile` — container instructions
- `README.md` — project documentation

## Step 1: Train the model

python train.py

## Step 2: Run the API locally

uvicorn main:app --reload

## Step 3: Build Docker image

docker build -t ml-fastapi-app 

## Step 4: Run Docker container

docker run -p 8000:8000 ml-fastapi-app

## Example request for /predict

json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

## in teams I 'll upload some photos that prove that is actually working 