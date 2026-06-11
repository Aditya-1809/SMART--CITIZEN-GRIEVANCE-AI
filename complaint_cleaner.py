import pandas as pd
import spacy

# -----------------------------
# LOAD NLP MODEL
# -----------------------------

nlp = spacy.load(
    "en_core_web_sm"
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
# CLEAN TEXT FUNCTION
# -----------------------------

def clean_text(
    text
):

    doc = nlp(
        text.lower()
    )

    cleaned_words = []

    for token in doc:

        if (
            not token.is_stop
            and
            not token.is_punct
        ):

            cleaned_words.append(
                token.lemma_
            )

    return " ".join(
        cleaned_words
    )

# -----------------------------
# APPLY CLEANING
# -----------------------------

df[
    "cleaned_text"
] = df[
    "complaint_text"
].apply(
    clean_text
)

# -----------------------------
# SHOW RESULT
# -----------------------------

print(
    "\n========================"
)

print(
    "CLEANED COMPLAINTS"
)

print(
    "========================\n"
)

print(
    df[
        [
            "complaint_text",
            "cleaned_text"
        ]
    ].head(10)
)