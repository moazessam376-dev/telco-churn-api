from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "churn"


def load(path: str | Path) -> pd.DataFrame:
    """Load and normalize the telco churn CSV."""
    frame = pd.read_csv(path)
    frame = frame.drop(columns=["customerID"])
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = frame["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return frame.drop(columns=["Churn"])


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split normalized data into stratified training and test sets."""
    features = df.drop(columns=[TARGET])
    target = df[TARGET]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
