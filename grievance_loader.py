import pandas as pd

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
# REMOVE EMPTY / DUPLICATES
# -----------------------------

all_complaints = (
    all_complaints
    .dropna()
    .drop_duplicates()
)

# -----------------------------
# SHOW DATA
# -----------------------------

print(
    "\n========================"
)

print(
    "FIRST 10 COMPLAINTS"
)

print(
    "========================\n"
)

print(
    all_complaints.head(10)
)

# -----------------------------
# DATASET SUMMARY
# -----------------------------

print(
    "\n========================"
)

print(
    "TOTAL COMPLAINTS"
)

print(
    "========================"
)

print(
    len(all_complaints)
)

print(
    "\n========================"
)

print(
    "DEPARTMENT COUNT"
)

print(
    "========================"
)

print(
    all_complaints[
        "department"
    ].value_counts()
)

print(
    "\n========================"
)

print(
    "SENTIMENT COUNT"
)

print(
    "========================"
)

print(
    all_complaints[
        "sentiment"
    ].value_counts()
)