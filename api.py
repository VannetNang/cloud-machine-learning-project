from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import boto3
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

# Load model
with open("diabetes_best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


# ----- Request Body Schema -----
class Patient(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float


# Access environment variables
bucket_name = os.getenv("DIABETES_BUCKET_NAME")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")

# ----- Optional: S3 logging -----
S3_BUCKET = bucket_name
s3_client = None
if S3_BUCKET:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        region_name="us-east-1",
    )


def log_to_s3(input_data, prediction):
    if not s3_client or not S3_BUCKET:
        return

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
    }

    key = f"predictions/{datetime.utcnow().date()}/{uuid.uuid4()}.json"

    import json

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(record),
        ContentType="application/json",
    )


# ----- Prediction Endpoint -----
@app.post("/predict")
def predict(patient: Patient):

    df = pd.DataFrame([patient.dict()])

    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[:, 1][0])

    label = "Diabetic" if pred == 1 else "Non-diabetic"

    result = {"prediction": int(pred), "label": label, "diabetes_probability": prob}

    log_to_s3(patient.dict(), result)

    return result
