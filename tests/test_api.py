import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn.data import CATEGORICAL, NUMERIC, TARGET, load
from churn.schemas import Customer
from churn.serve import create_app
from churn.train import save, train

DATA_PATH = Path(__file__).parents[1] / "data" / "telco_churn.csv"
VERSION = "api-test"


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Train and save one sample model for the API tests."""
    loaded = load(DATA_PATH)
    sampled, _ = train_test_split(
        loaded,
        train_size=1000,
        random_state=42,
        stratify=loaded[TARGET],
    )
    result = train(sampled)
    return save(result, tmp_path_factory.mktemp("api-model"), VERSION, data_path=DATA_PATH)


@pytest.fixture(scope="session")
def metadata(artifacts: tuple[Path, Path]) -> dict[str, object]:
    """Read the metadata written for the API test model."""
    _, metadata_path = artifacts
    return json.loads(metadata_path.read_text())


@pytest.fixture(scope="session")
def client(artifacts: tuple[Path, Path]) -> Iterator[TestClient]:
    """Provide a TestClient whose lifespan loads the saved test model."""
    model_path, _ = artifacts
    with TestClient(create_app(model_path=str(model_path))) as test_client:
        yield test_client


@pytest.fixture
def valid_customer() -> dict[str, object]:
    """Return a valid customer request body."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1572.0,
    }


def test_customer_dump_matches_training_columns(valid_customer: dict[str, object]) -> None:
    customer = Customer.model_validate(valid_customer)

    assert list(customer.model_dump()) == CATEGORICAL + NUMERIC


def test_health_reports_saved_model_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": VERSION}


def test_model_returns_saved_metadata(client: TestClient, metadata: dict[str, object]) -> None:
    response = client.get("/model")
    body = response.json()

    assert response.status_code == 200
    assert body["threshold"] == metadata["threshold"]
    assert body["metrics"] == metadata["metrics"]


def test_predict_returns_probability_and_threshold(
    client: TestClient,
    metadata: dict[str, object],
    valid_customer: dict[str, object],
) -> None:
    response = client.post("/predict", json=valid_customer)
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["churn"], bool)
    assert body["threshold"] == metadata["threshold"]
    assert body["model_version"] == VERSION


def test_predict_handles_missing_total_charges(
    client: TestClient,
    valid_customer: dict[str, object],
) -> None:
    customer_with_missing_total_charges = valid_customer | {"TotalCharges": None}

    response = client.post("/predict", json=customer_with_missing_total_charges)
    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1


def test_predict_is_deterministic(
    client: TestClient,
    valid_customer: dict[str, object],
) -> None:
    first = client.post("/predict", json=valid_customer)
    second = client.post("/predict", json=valid_customer)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["churn_probability"] == second.json()["churn_probability"]


def test_predict_rejects_invalid_category(
    client: TestClient,
    valid_customer: dict[str, object],
) -> None:
    invalid_customer = valid_customer | {"Contract": "Yearly"}

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_batch_predict_preserves_input_order(
    client: TestClient,
    valid_customer: dict[str, object],
) -> None:
    customers = [valid_customer | {"tenure": tenure} for tenure in (2, 24, 60)]

    batch_response = client.post("/predict/batch", json={"customers": customers})
    individual_probabilities = [
        client.post("/predict", json=customer).json()["churn_probability"]
        for customer in customers
    ]

    assert batch_response.status_code == 200
    batch_predictions = batch_response.json()["predictions"]
    assert len(batch_predictions) == 3
    assert [prediction["churn_probability"] for prediction in batch_predictions] == pytest.approx(
        individual_probabilities
    )


def test_batch_predict_rejects_more_than_1000_customers(
    client: TestClient,
    valid_customer: dict[str, object],
) -> None:
    response = client.post(
        "/predict/batch",
        json={"customers": [valid_customer] * 1001},
    )

    assert response.status_code == 422
