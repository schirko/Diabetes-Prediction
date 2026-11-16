"""The candidate models, each wrapped in a pipeline.

A pipeline bundles the preparation steps (fill gaps, rescale) with the model,
so cross-validation re-learns those steps on each training fold. That is what
keeps information from the test rows out of training.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def logistic():
    """Fill gaps with the training median, rescale, fit logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def candidates() -> dict:
    """Every model in the comparison, keyed by the name shown in tables."""
    return {
        "Always predict 'no diabetes'": DummyClassifier(strategy="prior"),
        "Glucose only (logistic)": logistic(),  # fitted on the Glucose column alone
        "Decision tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=SEED),
        ),
        "Logistic regression": logistic(),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
        ),
    }


def columns_for(name: str, all_columns: list) -> list:
    """Which input columns a named model is allowed to see."""
    return ["Glucose"] if name.startswith("Glucose only") else list(all_columns)
