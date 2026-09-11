from typing import Literal

from pydantic import BaseModel, Field


class Customer(BaseModel):
    """Validate one customer's features for a churn prediction."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    """Return a churn probability and the decision made at the model threshold."""

    churn_probability: float
    churn: bool
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    """Validate a batch of at most 1000 customer requests."""

    customers: list[Customer] = Field(max_length=1000)


class BatchResponse(BaseModel):
    """Return predictions in the same order as a batch request."""

    predictions: list[Prediction]
