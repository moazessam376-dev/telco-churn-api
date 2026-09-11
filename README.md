# telco-churn-api

A churn classifier trained with scikit-learn on the IBM Telco Customer Churn data and served by FastAPI. Training and serving share one pipeline object, so they cannot drift.

Work in progress. The decisions are in [docs/design.md](docs/design.md); the code follows them slice by slice, tests first.
