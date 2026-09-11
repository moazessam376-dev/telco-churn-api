# telco-churn-api

This project trains a scikit-learn pipeline that estimates whether a telecom customer will churn.
It serves the saved pipeline and its metadata through a FastAPI application.

## Data

The project uses the IBM Telco Customer Churn dataset described in [docs/design.md](docs/design.md).
The dataset has an Apache-2.0 license and is committed at `data/telco_churn.csv`.

## Train

Run training with a version for the artifacts:

```shell
uv run churn-train --data data/telco_churn.csv --out models --version <version>
```

This writes `models/churn-<version>.joblib` and its sibling metadata file,
`models/churn-<version>.json`.

## Serve

Start the API with the model artifact selected through `MODEL_PATH`:

```shell
MODEL_PATH=models/churn-<version>.joblib uv run uvicorn churn.serve:app
```

The endpoints are:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Return service status and model version. |
| GET | `/model` | Return the saved metadata JSON. |
| POST | `/predict` | Predict churn for one customer. |
| POST | `/predict/batch` | Predict churn for a list of customers. |

Example request to `POST /predict`:

```json
{
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
  "TotalCharges": 1572.0
}
```

Example response:

```json
{
  "churn_probability": 0.8044972342477199,
  "churn": true,
  "threshold": 0.5740846043098109,
  "model_version": "v1"
}
```

The response above is a real call against the v1 model with the first row of the dataset. The threshold is chosen to maximise F1 on cross-validated
predictions and is stored in the metadata. The API returns the probability, the yes or no decision
at that threshold, and the threshold itself, so a caller can apply a different cutoff.

## Metrics

Full training run on 2026-09-11, all 7,043 rows, `uv run churn-train --data data/telco_churn.csv --out models --version v1` (record: `models/churn-v1.json`):

| metric | value |
|---|---|
| chosen estimator | logistic_regression |
| cross-validated ROC-AUC, logistic regression | 0.8460 |
| cross-validated ROC-AUC, gradient boosting | 0.8377 |
| threshold (F1 on cross-validated probabilities) | 0.5741 |
| ROC-AUC, held-out test split | 0.8414 |
| PR-AUC, held-out test split | 0.6324 |
| precision at threshold | 0.5388 |
| recall at threshold | 0.7246 |
| Brier score | 0.1688 |


## Not in scope

Monitoring, retraining schedules, feature stores, Docker, authentication,
explanations per prediction.
