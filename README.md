# Ames Housing — LightGBM + MLflow
 
A regression pipeline predicting house sale prices on the Ames Housing
dataset, built to learn the MLOps layer around a model: experiment tracking, 
a model registry with a promotion gate, and
serving the registered model via FastAPI.
 
## Stack
 
- **Model:** LightGBM regression
- **Tracking / registry:** MLflow (SQLite-backed store, alias-based
  registry — `champion` points at the current best version)
- **Serving:** FastAPI, loads the `@champion` model at startup
## Project structure
 
```
.
├── src/features.py                     # prepare_X(): feature engineering + categorical casting
├── main.py                             # train_and_log() + promote_if_better()
├── serve.py                            # FastAPI app, /predict endpoint
├── config.py
├── predict_using_champion_model.py
├── test_served_prediction.ipynb
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```
 
## Usage

**0. Download data**

Dataset available at https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
 
**1. Train, log, and promote a model:**
 
```bash
python main.py
```
 
Each run:
- logs params + RMSE to MLflow (`mlflow ui` to inspect)
- registers a new version of model `ames-housing-lgbm`
- compares its RMSE against the current `champion`; promotes only if it's
  actually better, otherwise leaves the champion alone

**2. Serve the current champion:**
 
```bash
uvicorn serve:app --reload
```
 
**3. Call it:**
 
See `test_served_prediction.ipynb` — loads a row from the test set, sends
it to `/predict`, and compares the prediction against the actual
`SalePrice`.
 
## Design notes / things I'd flag in an interview
 
**Known simplification:** categorical columns are handled via native
pandas `category` dtype rather than a proper `sklearn.Pipeline` with a
fitted encoder. This works because `prepare_X` is shared, but it's still
two logically separate places (train + serve) applying the same casting
by convention rather than one fitted, serialized preprocessing object. A
`Pipeline` (encoder + model as one artifact) would remove that remaining
coupling entirely — noted as a follow-up, not fixed here.
 
# Next steps

## 1. `sklearn.Pipeline`
 
Bundle the categorical encoder + LightGBM model into one fitted, logged
object instead of relying on `prepare_X` being called consistently by both
training and serving. `prepare_X` shared across both is good, but it's
still two call sites relying on convention rather than one artifact that
carries its own preprocessing. A `Pipeline` removes that remaining
coupling entirely.
 
## 2. Pydantic validation on `/predict`
 
Right now a malformed or missing field in the request payload produces a
raw 500 traceback. A Pydantic request model gives a clean 422 response
with a useful error message instead, and documents the expected schema
for free via FastAPI's OpenAPI docs.
 
## 3. Airflow DAG
 
Wire `train_and_log()` + `promote_if_better()` into a scheduled DAG so
retraining happens automatically instead of via a manual script run. This
is mostly plumbing given existing Airflow experience — no new concepts,
just connecting what's already built.
 
## 4. Model signature, done properly
 
Now that `prepare_X` is stable, revisit `input_example`/signature logging
so MLflow can validate inputs at serve time and generate a proper example
payload, instead of dtype mismatches only surfacing as LightGBM
tracebacks.
 
## 5. A hosted MLflow tracking server
 
Move off local SQLite to a shared tracking server (with a proper artifact
store, e.g. S3/GCS) to simulate a team setting — one registry that
multiple people or pipelines read from and write to.