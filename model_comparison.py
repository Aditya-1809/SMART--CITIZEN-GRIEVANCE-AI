import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("all_departments.csv")


# -----------------------------------
# SHOW COLUMN NAMES
# -----------------------------------

print("\nDataset Columns:")
print(df.columns)


# -----------------------------------
# INPUT AND TARGET
# -----------------------------------

X = df["complaint_text"]
y = df["department"]


# -----------------------------------
# TF-IDF VECTORIZATION
# -----------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# MODELS TO COMPARE
# -----------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        )
}


print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================\n")


# -----------------------------------
# TRAIN + TEST EACH MODEL
# -----------------------------------

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"{name} Accuracy: {accuracy:.4f}\n"
    )


print("==============================")
print("COMPARISON COMPLETED")
print("==============================")