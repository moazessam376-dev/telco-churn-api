# telco-churn-api design

A churn model for a telecom customer base, trained with scikit-learn and served
by FastAPI. Given one customer's contract and usage fields, the service returns
the probability that the customer leaves and a yes or no at a chosen threshold.
Built 2026-09 as a practice project: "deploy a machine learning model into
production" in miniature, with the training and serving sides sharing one
pipeline object so they cannot drift.

## Data

IBM Telco Customer Churn, 7,043 rows, 21 columns, Apache-2.0, committed as
`data/telco_churn.csv` (SHA-256 prefix 16320c9c). One row per customer: tenure
in months, contract type, payment method, monthly and total charges, which
services they have, and whether they churned. About one customer in four has
churned, so the classes are imbalanced. It fits the owner's telecom
background, which gives the interview a story: which customers leave, and
what a retention team does with a probability.

## Decisions

Each decision: chosen / rejected / why here / what breaks if wrong.

1. One `Pipeline` holds preprocessing and the model. Rejected: preprocessing
   in the API code and a bare estimator in the artifact. With one object the
   serving side cannot encode a column differently from training, which is
   the classic way a deployed model quietly rots. If wrong: predictions drift
   from the offline metrics and nobody can see why.
2. Preprocessing is a `ColumnTransformer`: `OneHotEncoder(handle_unknown=
   "ignore")` for categorical columns, median imputation plus `StandardScaler`
   for numeric ones. Rejected: ordinal codes for categoricals (they invent an
   order), dropping rows with blanks (the blank `TotalCharges` rows are new
   customers, exactly the ones a churn model must handle). If wrong: an unseen
   category crashes serving, or new customers get nonsense scores.
3. Two candidates, chosen by cross-validated ROC-AUC: `LogisticRegression
   (class_weight="balanced")` and `HistGradientBoostingClassifier`. Rejected:
   a single model picked by taste, and a wider search. Two candidates show the
   selection mechanism without turning the project into a benchmark. If wrong:
   the weaker model ships; the metadata file says which one and why.
4. Evaluation on a held-out 20 percent split, stratified, with five-fold
   cross-validation on the training part for model selection and for the
   threshold. Rejected: choosing anything on the test split. The test split is
   touched once, for the numbers in the README. If wrong: the reported numbers
   are optimistic and an interviewer will ask exactly that.
5. Metrics: ROC-AUC and PR-AUC as threshold-free scores, plus precision and
   recall at the chosen threshold, and the Brier score for calibration.
   Rejected: accuracy, which is misleading at one-in-four churn (always saying
   "stays" scores about 0.73). If wrong: a model that never predicts churn
   looks good.
6. The threshold is chosen to maximise F1 on cross-validated predictions and
   is stored in the metadata; the API returns the probability and the yes or
   no at that threshold, and the threshold itself, so a caller can pick a
   different one. Rejected: a hard-coded 0.5. A retention team chooses recall
   over precision or the reverse depending on the cost of a call; the
   probability is the product, the threshold is a default. If wrong: the API
   hides the decision that matters most to the business.
7. The artifact is `joblib` plus a JSON metadata file: scikit-learn and Python
   versions, feature names, chosen estimator, threshold, metrics, data SHA-256
   and trained-at time. Rejected: pickle alone. A model file without its
   versions and its data hash cannot be trusted or reproduced. If wrong: a
   future scikit-learn silently loads a different object.
8. Serving loads the model once at startup through FastAPI's lifespan and
   validates every request with a Pydantic model whose categorical fields are
   `Literal` enums. Rejected: loading per request (slow) and free-text fields
   (a typo becomes a silent zero vector after one-hot encoding). If wrong: a
   wrong category returns a confident score for a customer that does not
   exist.
9. Endpoints: `GET /health`, `GET /model` (the metadata), `POST /predict` for
   one customer, `POST /predict/batch` for a list. Rejected: a single endpoint
   that accepts both shapes. Two shapes, two contracts, both documented by
   FastAPI's schema. If wrong: callers guess.
10. Training is reproducible in CI: the workflow retrains from the committed
    CSV and runs the tests, so the metrics table in the README is reproduced
    on every push. Rejected: committing the model artifact as the source of
    truth. The CSV plus the code is the source; the artifact is an output. If
    wrong: the README's numbers and the code drift apart.
11. No Docker. There is no Docker daemon on the build machine, so an image
    could not be verified, and an unverified Dockerfile is a claim. Rejected:
    shipping one anyway. If wrong: the README promises a container that may
    not build.
12. Python 3.12 through uv, `pyproject.toml`, `ruff`. Rejected: a
    requirements.txt with unpinned versions. If wrong: CI and the laptop train
    different models.

## Numbers

Rows 7,043. Test split 20 percent, stratified, `random_state` 42. Five folds.
Threshold from cross-validated F1. History: a training run writes
`models/churn-<version>.joblib` and `models/churn-<version>.json`.

## Not in scope

Monitoring, retraining schedules, feature stores, Docker, authentication,
explanations per prediction.
