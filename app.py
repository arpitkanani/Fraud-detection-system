import gc
import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from src.logger import logging
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils import ID_COLUMN, load_dataframe, load_json, merge_dataframes


app = FastAPI(
    title="Fraud Detection - IEEE CIS",
    description="Standard ML pipeline for fraud detection with CSV upload support.",
    version="3.0.0",
)
templates = Jinja2Templates(directory="templates")

MODEL_REPORT_PATH = os.path.join("artifacts", "model_report.json")
PREDICTION_OUTPUT_PATH = os.path.join("artifacts", "predictions", "prediction.csv")
PREVIEW_ROW_COUNT = 200
DEFAULT_RISK_PAGE_SIZE = 200
MAX_RISK_PAGE_SIZE = 1000
UPLOAD_TEMP_DIRECTORY = os.path.join("artifacts", "temp_uploads")
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def get_model_stats() -> dict[str, Any]:
    try:
        if os.path.exists(MODEL_REPORT_PATH):
            model_report = load_json(MODEL_REPORT_PATH)
            if isinstance(model_report, dict):
                return model_report.get("best_model", {})
        return {}
    except Exception:
        return {}


def validate_transaction_input(df) -> str | None:
    required_columns = {ID_COLUMN, "TransactionAmt", "card1"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        return (
            "Uploaded file is missing required columns: "
            f"{sorted(missing_columns)}."
        )
    return None


async def read_uploaded_csv(file: UploadFile, label: str):
    if file is None or file.filename == "":
        raise ValueError(f"Please upload the {label} CSV file.")

    os.makedirs(UPLOAD_TEMP_DIRECTORY, exist_ok=True)
    file_suffix = os.path.splitext(file.filename or "")[1] or ".csv"
    temp_file_descriptor, temp_file_path = tempfile.mkstemp(
        prefix=f"{label}_",
        suffix=file_suffix,
        dir=UPLOAD_TEMP_DIRECTORY,
    )
    os.close(temp_file_descriptor)

    try:
        with open(temp_file_path, "wb") as temp_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)

        dataframe = load_dataframe(temp_file_path)
        if dataframe.empty:
            raise ValueError(f"{label} file is empty.")
        return dataframe
    finally:
        await file.close()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def build_summary(results_df):
    return {
        "total": int(len(results_df)),
        "high": int((results_df["RiskFactor"] == "HIGH").sum()),
        "medium": int((results_df["RiskFactor"] == "MEDIUM").sum()),
        "no_risk": int((results_df["RiskFactor"] == "NO_RISK").sum()),
        "fraud_count": int((results_df["Prediction"] == 1).sum()),
        "fraud_pct": round(float((results_df["Prediction"] == 1).mean() * 100), 2),
        "avg_prob": round(float(results_df["FraudProbability"].mean()), 2),
        "preview_rows": min(PREVIEW_ROW_COUNT, len(results_df)),
    }


def is_reload_enabled() -> bool:
    return os.getenv("APP_RELOAD", "0").strip().lower() in {"1", "true", "yes"}


def get_prediction_output_display_path() -> str:
    return os.path.abspath(PREDICTION_OUTPUT_PATH)


def normalize_risk_factor(risk_factor: str) -> str:
    return str(risk_factor).strip().upper().replace(" ", "_")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_stats": get_model_stats(),
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "model_stats": get_model_stats(),
            "prediction_output_path": get_prediction_output_display_path(),
            "results": None,
            "summary": None,
            "error": None,
            "preview_note": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_upload(
    request: Request,
    upload_mode: str = Form(...),
    merged_file: UploadFile = File(default=None),
    tx_file: UploadFile = File(default=None),
    id_file: UploadFile = File(default=None),
):
    try:
        logging.info("PREDICT ENDPOINT CALLED WITH MODE: %s", upload_mode)
        if upload_mode == "merged":
            input_df = await read_uploaded_csv(merged_file, "merged")
        elif upload_mode == "separate":
            transaction_df = await read_uploaded_csv(tx_file, "transaction")
            identity_df = await read_uploaded_csv(id_file, "identity")
            input_df = merge_dataframes(transaction_df, identity_df, on=ID_COLUMN, how="left")
            del transaction_df
            del identity_df
            gc.collect()
        else:
            raise ValueError("Invalid upload mode selected.")

        validation_error = validate_transaction_input(input_df)
        if validation_error:
            raise ValueError(validation_error)

        prediction_pipeline = PredictionPipeline()
        results_df = prediction_pipeline.predict(input_df)
        preview_results = results_df.head(PREVIEW_ROW_COUNT).to_dict(orient="records")

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "model_stats": get_model_stats(),
                "prediction_output_path": get_prediction_output_display_path(),
                "results": preview_results,
                "summary": build_summary(results_df),
                "error": None,
                "preview_note": (
                    f"Showing top {min(PREVIEW_ROW_COUNT, len(results_df))} rows in the browser. "
                    "Use Download CSV for the complete prediction file."
                ),
            },
        )
    except Exception as error:
        message = str(error)
        logging.error("Prediction request failed: %s", message)
        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "model_stats": get_model_stats(),
                "prediction_output_path": get_prediction_output_display_path(),
                "results": None,
                "summary": None,
                "error": message,
                "preview_note": None,
            },
            status_code=400,
        )


@app.get("/download")
async def download_predictions():
    if not os.path.exists(PREDICTION_OUTPUT_PATH):
        return HTMLResponse("No prediction file found. Run a prediction first.", status_code=404)
    return FileResponse(
        PREDICTION_OUTPUT_PATH,
        media_type="text/csv",
        filename="prediction.csv",
    )


@app.get("/risk-transactions/{risk_factor}")
async def get_risk_transactions(
    risk_factor: str,
    page: int = 1,
    page_size: int = DEFAULT_RISK_PAGE_SIZE,
):
    normalized_risk_factor = normalize_risk_factor(risk_factor)
    allowed_risk_factors = {"HIGH", "MEDIUM", "NO_RISK"}
    if normalized_risk_factor not in allowed_risk_factors:
        raise HTTPException(status_code=400, detail="Invalid risk factor.")

    if not os.path.exists(PREDICTION_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="No prediction file found. Run a prediction first.")

    safe_page = max(page, 1)
    safe_page_size = min(max(page_size, 50), MAX_RISK_PAGE_SIZE)

    prediction_df = load_dataframe(PREDICTION_OUTPUT_PATH)
    if "RiskFactor" not in prediction_df.columns or ID_COLUMN not in prediction_df.columns:
        raise HTTPException(status_code=500, detail="Prediction output is missing required columns.")

    filtered_df = prediction_df.loc[
        prediction_df["RiskFactor"].astype(str).str.upper() == normalized_risk_factor,
        [ID_COLUMN, "FraudProbability", "Prediction", "RiskFactor"],
    ].reset_index(drop=True)

    total_count = len(filtered_df)
    total_pages = max((total_count + safe_page_size - 1) // safe_page_size, 1)
    safe_page = min(safe_page, total_pages)
    start_index = (safe_page - 1) * safe_page_size
    end_index = start_index + safe_page_size
    page_df = filtered_df.iloc[start_index:end_index].copy()

    return {
        "risk_factor": normalized_risk_factor,
        "display_risk_factor": normalized_risk_factor.replace("_", " "),
        "page": safe_page,
        "page_size": safe_page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "server_saved_path": get_prediction_output_display_path(),
        "transactions": page_df.to_dict(orient="records"),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_stats": get_model_stats(),
        "prediction_csv_exists": os.path.exists(PREDICTION_OUTPUT_PATH),
    }


if __name__ == "__main__":
    reload_enabled = is_reload_enabled()
    run_kwargs: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 5000,
        "reload": reload_enabled,
    }
    if reload_enabled:
        run_kwargs["reload_dirs"] = ["src", "templates"]
        run_kwargs["reload_excludes"] = ["artifacts/*", "logs/*", "notebooks/data/*"]

    uvicorn.run("app:app", **run_kwargs)
