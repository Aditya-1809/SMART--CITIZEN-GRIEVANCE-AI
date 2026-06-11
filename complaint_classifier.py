import pandas as pd
import joblib

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# -----------------------------
# LOAD DATASETS
# -----------------------------

roads_data = pd.read_csv(
    "roads.csv"
)

water_data = pd.read_csv(
    "water.csv"
)

electricity_data = pd.read_csv(
    "electricity.csv"
)

# -----------------------------
# MERGE DATASETS
# -----------------------------

all_complaints = pd.concat(
    [
        roads_data,
        water_data,
        electricity_data
    ],
    ignore_index=True
)

# -----------------------------
# CLEAN DATA
# -----------------------------

all_complaints = (
    all_complaints
    .dropna()
    .drop_duplicates()
)

# remove extra spaces
all_complaints["complaint_text"] = (
    all_complaints["complaint_text"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print(
    "Total Complaints:",
    len(all_complaints)
)

# -----------------------------
# FEATURES & TARGETS
# -----------------------------

X = all_complaints[
    "complaint_text"
]

department_target = (
    all_complaints[
        "department"
    ]
)

sentiment_target = (
    all_complaints[
        "sentiment"
    ]
)

# -----------------------------
# TEXT VECTORIZATION
# -----------------------------

text_encoder = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words="english",
    max_features=20000,
    min_df=1,
    max_df=0.90,
    sublinear_tf=True
)

X_vectorized = (
    text_encoder.fit_transform(X)
)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

(
    X_train,
    X_test,
    dept_train,
    dept_test,
    sent_train,
    sent_test
) = train_test_split(
    X_vectorized,
    department_target,
    sentiment_target,
    test_size=0.10,
    random_state=42,
    stratify=sentiment_target
)

# -----------------------------
# DEPARTMENT MODEL
# -----------------------------

department_model = LogisticRegression(
    max_iter=10000,
    C=8,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)

department_model.fit(
    X_train,
    dept_train
)

department_prediction = (
    department_model.predict(X_test)
)

department_accuracy = (
    accuracy_score(
        dept_test,
        department_prediction
    )
)

# -----------------------------
# SENTIMENT MODEL
# -----------------------------

sentiment_model = LogisticRegression(
    max_iter=10000,
    C=8,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)

sentiment_model.fit(
    X_train,
    sent_train
)

sentiment_prediction = (
    sentiment_model.predict(X_test)
)

sentiment_accuracy = (
    accuracy_score(
        sent_test,
        sentiment_prediction
    )
)

# -----------------------------
# SAVE FILES
# -----------------------------

joblib.dump(
    department_model,
    "department_model.pkl"
)

joblib.dump(
    sentiment_model,
    "sentiment_model.pkl"
)

joblib.dump(
    text_encoder,
    "text_encoder.pkl"
)

# -----------------------------
# RESULTS
# -----------------------------

print("\n========================")
print("DEPARTMENT ACCURACY")
print("========================")

print(
    round(
        department_accuracy * 100,
        2
    ),
    "%"
)

print("\n========================")
print("SENTIMENT ACCURACY")
print("========================")

print(
    round(
        sentiment_accuracy * 100,
        2
    ),
    "%"
)

print("\n========================")
print("CLASSIFICATION REPORT")
print("========================")

print(
    classification_report(
        sent_test,
        sentiment_prediction
    )
)

print("\n========================")
print("FILES SAVED SUCCESSFULLY")
print("========================")

print("department_model.pkl")
print("sentiment_model.pkl")
print("text_encoder.pkl")