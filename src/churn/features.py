from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """Build the shared categorical and numeric preprocessing transformer."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", numeric, NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
