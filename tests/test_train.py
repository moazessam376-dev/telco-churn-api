import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import train_test_split

from churn.data import TARGET, load, split
from churn.train import save, train

DATA_PATH = Path(__file__).parents[1] / "data" / "telco_churn.csv"


def test_train_and_save_sample_artifacts(tmp_path: Path) -> None:
    loaded = load(DATA_PATH)
    sampled, _ = train_test_split(
        loaded,
        train_size=500,
        random_state=42,
        stratify=loaded[TARGET],
    )

    result = train(sampled)
    model_path, metadata_path = save(result, tmp_path, "test", data_path=DATA_PATH)

    assert model_path == tmp_path / "churn-test.joblib"
    assert metadata_path == tmp_path / "churn-test.json"
    assert model_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert {
        "sklearn_version",
        "python_version",
        "feature_names",
        "chosen",
        "threshold",
        "metrics",
        "data_sha256",
        "trained_at",
        "version",
    } <= metadata.keys()
    assert metadata["sklearn_version"] == sklearn.__version__
    assert metadata["feature_names"] == result.feature_names
    assert metadata["chosen"] == result.chosen
    assert 0 <= metadata["threshold"] <= 1
    assert set(metadata["metrics"]) == {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "brier",
        "cv_roc_auc",
    }
    assert set(metadata["metrics"]["cv_roc_auc"]) == set(result.metrics["cv_roc_auc"])
    assert len(metadata["feature_names"]) > 0
    assert metadata["data_sha256"] == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert metadata["version"] == "test"
    assert result.metrics["roc_auc"] > 0.7

    pipeline = joblib.load(model_path)
    _, X_test, _, _ = split(sampled)
    probabilities = pipeline.predict_proba(X_test.iloc[[0]])

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1
