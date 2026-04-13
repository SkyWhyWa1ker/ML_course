from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()


model = joblib.load("model.joblib")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: IrisFeatures):
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    return {"prediction": int(prediction)}