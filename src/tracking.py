"""Milestone 10 (E27): record each evaluation run with MLflow, so runs can be compared over time.

MLflow writes to a small local database (`mlflow.db`) and a `mlartifacts/` folder, both ignored by Git. Nothing here
changes any result; if MLflow is not installed, or logging fails, the evaluation still finishes and says so.
"""
import json
import subprocess

from .data import ROOT

EXPERIMENT = "diabetes-prediction"
DB = "mlflow.db"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def headline(metrics: dict) -> dict:
    """The few numbers worth comparing from run to run."""
    cur = metrics["current"]
    return {
        "auc": cur["oof_auc"], "auc_low": cur["oof_auc_ci"][0], "auc_high": cur["oof_auc_ci"][1],
        "calibration_slope": cur["calibration"]["slope"],
        "sensitivity_at_cutoff": metrics["m5"]["published"]["counts"]["sensitivity"],
        "flagged_share_at_cutoff": metrics["m5"]["published"]["counts"]["flagged_share"],
        "external_auc": metrics["m8"]["auc"]["value"],
        "external_mean_predicted": metrics["m8"]["calibration"]["mean_predicted"],
        "conformal_coverage_mean": metrics["m7"]["coverage"]["coverage_mean"],
    }


def log_run(metrics: dict, files: list) -> str:
    """Log one evaluation run: settings, headline numbers and the report files. Returns the run id, or "" if skipped."""
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed, so this run was not recorded (pip install mlflow).")
        return ""
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / DB).as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.log_params({"features": ",".join(metrics["current"]["features"]), "cv_splits": metrics["cv"]["splits"],
                           "cv_repeats": metrics["cv"]["repeats"], "n_women": metrics["n"], "cutoff": metrics["m5"]["published"]["cutoff"],
                           "cost_ratio": metrics["m5"]["published"]["ratio"], "git_commit": _git_commit()})
        mlflow.log_metrics(headline(metrics))
        for f in files:
            if f.exists():
                mlflow.log_artifact(str(f))
        return run.info.run_id
