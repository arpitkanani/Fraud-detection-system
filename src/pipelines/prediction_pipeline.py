import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    align_dataframe_to_columns,
    coerce_columns_to_numeric,
    extract_expected_model_columns,
    feature_engineering,
    get_default_prediction_input_columns,
    load_json,
    load_object,
    map_risk_factor,
    normalize_column_names,
    save_dataframe,
    save_prediction_preview_image,
)


@dataclass
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    model_path: str = os.path.join("artifacts", "model.pkl")
    feature_assets_path: str = os.path.join(
        "artifacts", "feature_engineering", "feature_assets.pkl"
    )
    schema_file_path: str = os.path.join("artifacts", "feature_engineering", "schema.json")
    prediction_output_path: str = os.path.join(
        "artifacts", "predictions", "prediction.csv"
    )
    prediction_preview_chart_path: str = os.path.join(
        "artifacts", "predictions", "prediction_summary.png"
    )


class PredictionPipeline:
    def __init__(self) -> None:
        self.config = PredictionPipelineConfig()

    def _load_feature_assets(self) -> dict | None:
        if os.path.exists(self.config.feature_assets_path):
            return load_object(self.config.feature_assets_path)
        return None

    def _load_expected_columns(self, preprocessor) -> list[str]:
        if os.path.exists(self.config.schema_file_path):
            schema = load_json(self.config.schema_file_path)
            if isinstance(schema, dict) and "engineered_input_columns" in schema:
                return list(schema["engineered_input_columns"])
        return extract_expected_model_columns(preprocessor)

    def _load_numeric_columns(self) -> list[str]:
        if os.path.exists(self.config.schema_file_path):
            schema = load_json(self.config.schema_file_path)
            if isinstance(schema, dict) and "numeric_columns" in schema:
                return list(schema["numeric_columns"])
        return []

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("=" * 80)
            logging.info("PREDICTION PIPELINE STARTED")

            normalized_df = normalize_column_names(input_df)
            if ID_COLUMN not in normalized_df.columns:
                raise ValueError(f"{ID_COLUMN} column is required for prediction.")

            transaction_ids = normalized_df[ID_COLUMN].reset_index(drop=True)
            feature_df = normalized_df.drop(columns=[TARGET_COLUMN], errors="ignore")
            feature_df = align_dataframe_to_columns(
                feature_df,
                get_default_prediction_input_columns(),
                np.nan,
            )

            preprocessor = load_object(self.config.preprocessor_path)
            model = load_object(self.config.model_path)
            feature_assets = self._load_feature_assets()

            engineered_df = feature_engineering(feature_df, feature_assets=feature_assets)
            expected_model_columns = self._load_expected_columns(preprocessor)
            model_input_df = align_dataframe_to_columns(
                engineered_df.drop(columns=[ID_COLUMN, TIME_COLUMN], errors="ignore"),
                expected_model_columns,
                np.nan,
            )
            model_input_df = coerce_columns_to_numeric(
                model_input_df,
                self._load_numeric_columns(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                transformed_features = preprocessor.transform(model_input_df)
            fraud_probability = model.predict_proba(transformed_features)[:, 1]
            predictions = model.predict(transformed_features).astype(int)

            results_df = pd.DataFrame(
                {
                    "SNo": np.arange(1, len(transaction_ids) + 1),
                    ID_COLUMN: transaction_ids,
                    "Prediction": predictions,
                    "FraudProbability": np.round(fraud_probability * 100, 2),
                    "RiskFactor": [map_risk_factor(probability) for probability in fraud_probability],
                }
            )

            save_dataframe(results_df, self.config.prediction_output_path)
            save_prediction_preview_image(
                results_df,
                self.config.prediction_preview_chart_path,
            )

            logging.info("PREDICTION PIPELINE COMPLETED. Shape: %s", results_df.shape)
            logging.info("=" * 80)
            return results_df
        except Exception as error:
            raise CustomException(error, sys)  # type: ignore[arg-type]
