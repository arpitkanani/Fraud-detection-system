# FraudShield: IEEE-CIS Fraud Detection

FraudShield is an end-to-end fraud detection project built on the Kaggle IEEE-CIS Fraud Detection dataset. The repository includes a training pipeline, feature engineering, model selection across multiple gradient boosting models, and a FastAPI web app for batch CSV prediction.

## Features

- Time-based train/validation split to reduce leakage
- Feature engineering for transaction, card, email, device, address, identity, and V-series features
- Sklearn preprocessing pipeline with ordinal encoding, frequency encoding, and numeric imputation
- Model comparison across LightGBM, XGBoost, and CatBoost
- FastAPI UI for merged or separate CSV uploads
- Saved prediction CSV, summary chart, and training artifacts under `artifacts/`

## Project Structure

```text
Project-One/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipelines/
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── notebooks/
│   ├── 01_EDA_transaction.ipynb
│   ├── Identity_EDA.ipynb
│   ├── correlation_analysis.ipynb
│   ├── EDA_output_Graphs/
│   └── data/                  # local CSVs, git-ignored
├── prompts/
├── artifacts/                 # generated, git-ignored
├── logs/                      # generated, git-ignored
└── catboost_info/             # generated, git-ignored
```

## Requirements

- Python 3.10 or newer
- `pip`
- Kaggle IEEE-CIS Fraud Detection CSV files

Install dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset Setup

Download the dataset from Kaggle and place these files inside `notebooks/data/`:

```text
notebooks/data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

These raw CSV files are intentionally excluded from git.

## Train the Model

Run the training pipeline:

```bash
python -m src.pipelines.training_pipeline
```

This pipeline:

1. Loads and merges transaction and identity data
2. Performs a time-based train/validation split
3. Applies feature engineering and preprocessing
4. Trains and compares LightGBM, XGBoost, and CatBoost
5. Saves the best model and evaluation artifacts

Typical generated files:

```text
artifacts/
├── raw.parquet
├── train.parquet
├── val.parquet
├── test.parquet
├── preprocessor.pkl
├── model.pkl
├── model_report.json
├── roc_curve.png
├── confusion_matrix.png
├── feature_engineering/
│   ├── feature_assets.pkl
│   └── schema.json
└── predictions/
    ├── prediction.csv
    └── prediction_summary.png
```

## Run the App

Start the FastAPI app:

```bash
python app.py
```

The app runs on `http://127.0.0.1:5000`.

If you prefer running with Uvicorn directly:

```bash
uvicorn app:app --host 127.0.0.1 --port 5000 --reload
```

## Prediction Flow

The `/predict` page supports two upload modes:

- A single merged CSV containing transaction and identity columns together
- Two separate CSV files, one transaction file and one identity file

For prediction, the app requires `TransactionID`, `TransactionAmt`, and `card1` in the input transaction data.

Prediction results are saved to:

```text
artifacts/predictions/prediction.csv
```

Risk labels are assigned as:

- `HIGH` for probability `>= 0.70`
- `MEDIUM` for probability `>= 0.40`
- `NO_RISK` otherwise

## API Endpoints

- `GET /` - landing page with model summary
- `GET /predict` - upload form
- `POST /predict` - run batch prediction
- `GET /download` - download the latest prediction CSV
- `GET /risk-transactions/{risk_factor}` - paginated transactions for `HIGH`, `MEDIUM`, or `NO_RISK`
- `GET /health` - service and artifact status
- `GET /docs` - auto-generated Swagger UI

## Notes

- Train the model before using prediction routes, or provide the required files in `artifacts/`.
- Column names are normalized internally, so identity fields like `id-01` and `id_01` are handled consistently.
- Generated folders such as `artifacts/`, `logs/`, `catboost_info/`, and local dataset files are ignored by git.
