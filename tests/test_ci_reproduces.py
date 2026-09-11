import json
from pathlib import Path

import pytest

MODELS_PATH = Path(__file__).parents[1] / "models"
CI_METADATA_PATH = MODELS_PATH / "churn-ci.json"
RECORDED_METADATA_PATH = MODELS_PATH / "churn-v1.json"


@pytest.mark.skipif(
    not CI_METADATA_PATH.exists(),
    reason="CI model metadata has not been generated",
)
def test_ci_training_reproduces_recorded_metadata() -> None:
    ci_metadata = json.loads(CI_METADATA_PATH.read_text())
    recorded_metadata = json.loads(RECORDED_METADATA_PATH.read_text())

    assert ci_metadata["chosen"] == recorded_metadata["chosen"]

    metric_names = ("roc_auc", "pr_auc", "precision", "recall", "brier")
    for metric_name in metric_names:
        assert ci_metadata["metrics"][metric_name] == pytest.approx(
            recorded_metadata["metrics"][metric_name],
            abs=1e-3,
            rel=0,
        )

    for estimator_name in ("logistic_regression", "hist_gradient_boosting"):
        assert ci_metadata["metrics"]["cv_roc_auc"][estimator_name] == pytest.approx(
            recorded_metadata["metrics"]["cv_roc_auc"][estimator_name],
            abs=1e-3,
            rel=0,
        )

    assert ci_metadata["threshold"] == pytest.approx(
        recorded_metadata["threshold"],
        abs=1e-3,
        rel=0,
    )
