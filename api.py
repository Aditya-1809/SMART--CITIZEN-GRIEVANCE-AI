from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# -----------------------------
# CREATE APP
# -----------------------------

app = FastAPI(
    title="Smart Citizen Grievance AI",
    description="Predict Department and Sentiment",
    version="1.0"
)

# -----------------------------
# LOAD MODELS
# -----------------------------

department_model = joblib.load(
    "department_model.pkl"
)

sentiment_model = joblib.load(
    "sentiment_model.pkl"
)

text_encoder = joblib.load(
    "text_encoder.pkl"
)

# -----------------------------
# INPUT FORMAT
# -----------------------------

class ComplaintInput(
    BaseModel
):
    complaint: str

# -----------------------------
# HOME
# -----------------------------

@app.get("/")
def home():

    return {
        "message":
        "Smart Citizen Grievance AI Running"
    }

# -----------------------------
# PREDICT
# -----------------------------

@app.post("/predict")
def predict(
    data: ComplaintInput
):

    complaint_text = (
        data.complaint
        .lower()
        .strip()
    )

    complaint_vector = (
        text_encoder.transform(
            [complaint_text]
        )
    )

    department = (
        department_model.predict(
            complaint_vector
        )[0]
    )

    sentiment = (
        sentiment_model.predict(
            complaint_vector
        )[0]
    )

    return {
        "complaint":
        complaint_text,

        "department":
        department,

        "sentiment":
        sentiment
    }