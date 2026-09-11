import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI

from churn.data import CATEGORICAL, NUMERIC
from churn.schemas import BatchRequest, BatchResponse, Customer, Prediction

FEATURE_COLUMNS = CATEGORICAL + NUMERIC


def create_app(model_path: str | None = None) -> FastAPI:
    """Create a FastAPI application that loads one saved model at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured_path = model_path if model_path is not None else os.environ.get("MODEL_PATH")
        if configured_path is None:
            raise RuntimeError("A model path is required. Set MODEL_PATH or pass model_path.")

        saved_model_path = Path(configured_path)
        metadata_path = saved_model_path.with_suffix(".json")
        app.state.pipeline = joblib.load(saved_model_path)
        app.state.metadata = json.loads(metadata_path.read_text())
        yield

    app = FastAPI(lifespan=lifespan)

    def predict_customers(customers: list[Customer]) -> list[Prediction]:
        if not customers:
            return []

        frame = pd.DataFrame(
            [customer.model_dump() for customer in customers],
            columns=FEATURE_COLUMNS,
        )
        probabilities = app.state.pipeline.predict_proba(frame)[:, 1]
        threshold = float(app.state.metadata["threshold"])
        model_version = str(app.state.metadata["version"])
        return [
            Prediction(
                churn_probability=float(probability),
                churn=bool(probability >= threshold),
                threshold=threshold,
                model_version=model_version,
            )
            for probability in probabilities
        ]

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the model loaded and identify its version."""
        return {"status": "ok", "model_version": str(app.state.metadata["version"])}

    @app.get("/model")
    def model() -> dict[str, Any]:
        """Return the saved model metadata without modification."""
        return app.state.metadata

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer) -> Prediction:
        """Predict churn for one validated customer."""
        return predict_customers([customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest) -> BatchResponse:
        """Predict churn for a validated batch in input order."""
        return BatchResponse(predictions=predict_customers(request.customers))

    return app


app = create_app()
