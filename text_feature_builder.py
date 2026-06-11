import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
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

df = pd.concat(
    [
        roads_data,
        water_data,
        electricity_data
    ],
    ignore_index=True
)

# -----------------------------
# REMOVE EMPTY VALUES
# -----------------------------

df = (
    df
    .dropna()
    .drop_duplicates()
)

# -----------------------------
# TF-IDF FEATURE BUILDING
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=12000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df[
        "complaint_text"
    ]
)

# -----------------------------
# RESULTS
# -----------------------------

print(
    "\n========================"
)

print(
    "TEXT FEATURE MATRIX"
)

print(
    "========================"
)

print(
    "Matrix Shape:",
    X.shape
)

print(
    "\n========================"
)

print(
    "TOP FEATURES"
)

print(
    "========================"
)

features = (
    vectorizer
    .get_feature_names_out()
)

print(
    features[:50]
)