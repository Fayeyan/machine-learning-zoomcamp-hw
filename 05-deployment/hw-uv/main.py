import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "pipeline_v2.bin" if os.path.exists("pipeline_v2.bin") else "pipeline_v1.bin"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/predict")

def predict(client: Client):
    x = {
        "lead_source": client.lead_source,
        "number_of_courses_viewed": client.number_of_courses_viewed,
        "annual_income": client.annual_income,
    }
    p = model.predict_proba([x])[0, 1]
    return {"probability": float(p)}