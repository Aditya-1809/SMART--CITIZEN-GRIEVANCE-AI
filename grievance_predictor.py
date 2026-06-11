import joblib

# -----------------------------
# LOAD TRAINED MODELS
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
# TAKE USER INPUT
# -----------------------------

user_input = input(
    "Enter complaint: "
).strip().lower()

# -----------------------------
# VALIDATION
# -----------------------------

if not user_input:

    print(
        "Complaint cannot be empty."
    )

else:

    # -----------------------------
    # CONVERT TEXT
    # -----------------------------

    user_vector = (
        text_encoder.transform(
            [user_input]
        )
    )

    # -----------------------------
    # PREDICT
    # -----------------------------

    department_prediction = (
        department_model.predict(
            user_vector
        )[0]
    )

    sentiment_prediction = (
        sentiment_model.predict(
            user_vector
        )[0]
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------

    print(
        "\n========================"
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "========================"
    )

    print(
        "Complaint:",
        user_input
    )

    print(
        "Department:",
        department_prediction
    )

    print(
        "Sentiment:",
        sentiment_prediction
    )