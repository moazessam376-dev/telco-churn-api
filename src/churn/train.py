import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from churn.data import TARGET, load, split
from churn.features import build_preprocessor

Metrics = dict[str, float | dict[str, float]]


@dataclass
class TrainResult:
    pipeline: Pipeline
    chosen: str
    threshold: float
    metrics: Metrics
    feature_names: list[str]


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    if thresholds.size == 0:
        return 0.5
    best_index = int(np.argmax(f1_scores))
    return float(np.clip(thresholds[best_index], 0.0, 1.0))


def train(df: pd.DataFrame, random_state: int = 42) -> TrainResult:
    """Train, select, threshold, and evaluate a churn classifier."""
    X_train, X_test, y_train, y_test = split(df, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    candidates = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }
    pipelines = {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("estimator", estimator),
            ]
        )
        for name, estimator in candidates.items()
    }

    cv_roc_auc = {
        name: float(cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc").mean())
        for name, pipeline in pipelines.items()
    }
    chosen = max(cv_roc_auc, key=cv_roc_auc.get)
    chosen_pipeline = pipelines[chosen]

    oof_probabilities = cross_val_predict(
        chosen_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    threshold = _choose_threshold(y_train, oof_probabilities)

    chosen_pipeline.fit(X_train, y_train)
    test_probabilities = chosen_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = test_probabilities >= threshold
    metrics: Metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "pr_auc": float(average_precision_score(y_test, test_probabilities)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_test, test_probabilities)),
        "cv_roc_auc": cv_roc_auc,
    }
    feature_names = chosen_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    return TrainResult(
        pipeline=chosen_pipeline,
        chosen=chosen,
        threshold=threshold,
        metrics=metrics,
        feature_names=feature_names,
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(
    result: TrainResult,
    out_dir: str | Path,
    version: str,
    data_path: str | Path | None = None,
    data_sha256: str | None = None,
) -> tuple[Path, Path]:
    """Save the fitted pipeline and its reproducibility metadata."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"churn-{version}.joblib"
    metadata_path = output_dir / f"churn-{version}.json"

    if data_sha256 is None and data_path is not None:
        data_sha256 = _sha256(data_path)

    joblib.dump(result.pipeline, model_path)
    metadata: dict[str, Any] = {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_names": result.feature_names,
        "chosen": result.chosen,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "data_sha256": data_sha256,
        "trained_at": datetime.now(UTC).isoformat(),
        "version": version,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a telco churn model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for model artifacts.")
    parser.add_argument(
        "--version",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        help="Artifact version, defaulting to the current timestamp.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional stratified number of rows to train on.",
    )
    args = parser.parse_args()

    frame = load(args.data)
    if args.sample is not None:
        if args.sample <= 0 or args.sample > len(frame):
            parser.error(f"--sample must be between 1 and {len(frame)}")
        frame, _ = train_test_split(
            frame,
            train_size=args.sample,
            random_state=42,
            stratify=frame[TARGET],
        )

    result = train(frame)
    model_path, metadata_path = save(
        result,
        args.out,
        args.version,
        data_sha256=_sha256(args.data),
    )

    print(f"chosen: {result.chosen}")
    print(f"threshold: {result.threshold:.4f}")
    for name, value in result.metrics.items():
        if isinstance(value, dict):
            print(f"{name}: " + ", ".join(f"{key}={score:.4f}" for key, score in value.items()))
        else:
            print(f"{name}: {value:.4f}")
    print(f"model: {model_path}")
    print(f"metadata: {metadata_path}")
