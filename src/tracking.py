"""
tracking.py — one place that decides where experiment runs are logged.

Every training script calls setup_mlflow() before logging. Centralizing it
means the backend URI and experiment name can never disagree between scripts —
the same single-source-of-truth idea we used for features.py, applied to the
experiment config. If you later move to a hosted MLflow server, you change the
URI here once instead of hunting through every trainer.

Backend note: recent MLflow deprecated the plain ./mlruns file store for the
UI server, so we log to a local SQLite database instead. View the UI with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "like-a-bosch"


def setup_mlflow() -> None:
    """Point MLflow at the local SQLite store and select the experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
