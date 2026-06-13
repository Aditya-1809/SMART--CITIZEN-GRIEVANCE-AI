# Smart Citizen Grievance AI

## Overview

Smart Citizen Grievance AI is a Machine Learning-based complaint classification system that automatically classifies citizen complaints into the correct government department and predicts complaint sentiment severity.

The system predicts:

### Departments
- Roads
- Water
- Electricity

### Sentiments
- Positive
- Neutral
- Negative
- Critical

The project uses **Machine Learning**, **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, **Logistic Regression**, and **FastAPI** for prediction.

---

## Features

- Department Classification
- Complaint Sentiment Prediction
- FastAPI REST API
- Swagger API Documentation (`/docs`)
- Machine Learning Model Training
- Text Preprocessing
- TF-IDF Feature Extraction
- Data Visualization using EDA
- Real-time Complaint Prediction

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook
- Matplotlib

---

## Project Structure

```text
smart-citizen-grievance-ai/

│── api.py
│── complaint_classifier.py
│── grievance_predictor.py
│── complaint_cleaner.py
│── grievance_loader.py
│── text_feature_builder.py
│── roads.csv
│── water.csv
│── electricity.csv
│── department_model.pkl
│── sentiment_model.pkl
│── text_encoder.pkl
│── roads_eda.ipynb
│── requirements.txt
│── README.md
```

---

## Dataset Information

The model is trained using complaint datasets from three government departments.

### Roads Department

Handles complaints related to:

- Potholes
- Road damage
- Broken dividers
- Unsafe roads
- Traffic-related road issues

### Water Department

Handles complaints related to:

- Water shortage
- Pipeline leakage
- Dirty water
- Low water pressure
- Water contamination

### Electricity Department

Handles complaints related to:

- Power cuts
- Transformer blast
- Voltage fluctuation
- Electricity outage
- Meter issues

---

## Machine Learning Workflow

1. Load complaint datasets
2. Merge all department datasets
3. Clean and preprocess complaint text
4. Convert text into numerical features using TF-IDF
5. Train Logistic Regression models
6. Predict:
   - Department
   - Sentiment
7. Save trained models using Joblib

---

## Model Accuracy

### Department Accuracy
**98% – 100%**

### Sentiment Accuracy
**88% – 89%**

---

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Train Model

Run the training model:

```bash
python complaint_classifier.py
```

---

## Run Prediction Script

Run the complaint predictor:

```bash
python grievance_predictor.py
```

---

## Run FastAPI Server

Start the API server:

```bash
uvicorn api:app --reload
```

Open Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Example

### Input

```json
{
    "complaint": "Transformer blast near locality"
}
```

### Output

```json
{
    "complaint": "transformer blast near locality",
    "department": "Electricity",
    "sentiment": "Critical"
}
```

---

## Example Test Complaints

### Test 1 — Electricity Complaint

**Input:**

```text
Transformer blast near locality
```

**Expected Output:**

```text
Department: Electricity
Sentiment: Critical
```

---

### Test 2 — Water Complaint

**Input:**

```text
No drinking water available
```

**Expected Output:**

```text
Department: Water
Sentiment: Critical
```

---

### Test 3 — Roads Complaint

**Input:**

```text
Huge potholes near station
```

**Expected Output:**

```text
Department: Roads
Sentiment: Critical
```

---

### Test 4 — Positive Complaint

**Input:**

```text
Road repaired successfully
```

**Expected Output:**

```text
Department: Roads
Sentiment: Positive
```

---

## Future Improvements

- Add more departments
- Improve model accuracy
- Add frontend dashboard
- Cloud deployment
- Real-time complaint monitoring
- More NLP improvements

---

## Author

**Aditya Bhandarakar**
**Pooja Gupta**
