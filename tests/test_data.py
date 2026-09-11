from pathlib import Path

import pandas as pd

from churn.data import CATEGORICAL, NUMERIC, TARGET, load

DATA_PATH = Path(__file__).parents[1] / "data" / "telco_churn.csv"


def test_load_converts_blank_total_charges_to_missing() -> None:
    loaded = load(DATA_PATH)

    assert loaded["TotalCharges"].isna().any()


def test_load_maps_churn_to_integer_binary_column() -> None:
    loaded = load(DATA_PATH)

    assert "Churn" not in loaded.columns
    assert pd.api.types.is_integer_dtype(loaded[TARGET])
    assert set(loaded[TARGET].unique()) == {0, 1}


def test_load_has_exact_feature_and_target_columns() -> None:
    loaded = load(DATA_PATH)

    assert set(loaded.columns) == set(CATEGORICAL + NUMERIC + [TARGET])
