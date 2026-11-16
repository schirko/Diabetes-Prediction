"""Load the Pima diabetes data and mark the hidden missing values.

The file has no blank cells, but five measurements use 0 to mean
"not recorded" (nobody has a blood pressure or a BMI of zero).
Everything downstream starts from load_data(), so that fix happens once.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes.csv"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# A zero in these columns is physically impossible, so it means "missing".
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Plain-language names and units, used by the charts and the web page.
LABELS = {
    "Pregnancies": ("Pregnancies", "count"),
    "Glucose": ("Glucose, 2 hours after a sugar drink", "mg/dL"),
    "BloodPressure": ("Diastolic blood pressure", "mm Hg"),
    "SkinThickness": ("Triceps skinfold thickness", "mm"),
    "Insulin": ("Insulin, 2 hours after a sugar drink", "mu U/mL"),
    "BMI": ("Body mass index", "kg/m2"),
    "DiabetesPedigreeFunction": ("Family-history score", "score"),
    "Age": ("Age", "years"),
}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """The file exactly as published, zeros and all."""
    return pd.read_csv(path)


def load_data(path: Path = DATA_PATH):
    """Return (X, y) with impossible zeros replaced by NaN.

    Nothing is filled in here. Filling happens inside the model pipeline,
    so it is learned from training rows only.
    """
    df = load_raw(path)
    X = df[FEATURES].astype(float)
    X[ZERO_MEANS_MISSING] = X[ZERO_MEANS_MISSING].replace(0, np.nan)
    y = df[TARGET].astype(int)
    return X, y


def missing_summary(path: Path = DATA_PATH) -> pd.DataFrame:
    """How many values are missing in each column once zeros are read correctly."""
    X, _ = load_data(path)
    out = pd.DataFrame({"missing": X.isna().sum(), "percent": (X.isna().mean() * 100).round(1)})
    return out.sort_values("missing", ascending=False)
