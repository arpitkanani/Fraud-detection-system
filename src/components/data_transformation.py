import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    FrequencyEncoder,
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    build_feature_engineering_assets,
    convert_to_numeric_frame,
    convert_to_string_frame,
    extract_expected_model_columns,
    feature_engineering,
    get_freq_encode_cols,
    get_label_encode_cols,
    get_passthrough_cols,
    load_dataframe,
    save_json,
    save_object,
    validate_columns,
)


def convert_to_string(X):
    return convert_to_string_frame(X)


def convert_to_numeric(X):
    return convert_to_numeric_frame(X)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    feature_assets_file_path: str = os.path.join(
        "artifacts", "feature_engineering", "feature_assets.pkl"
    )
    schema_file_path: str = os.path.join(
        "artifacts", "feature_engineering", "schema.json"
    )


class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    def get_data_transformer_object(
        self,
        label_cols: list[str],
        freq_cols: list[str],
        passthrough_cols: list[str],
    ) -> ColumnTransformer:
        try:
            logging.info("Building sklearn ColumnTransformer")

            label_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "to_string",
                        FunctionTransformer(convert_to_string, validate=False),
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            dtype=np.float64,
                        ),
                    ),
                ]
            )

            frequency_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "to_string",
                        FunctionTransformer(convert_to_string, validate=False),
                    ),
                    ("frequency_encoder", FrequencyEncoder()),
                ]
            )

            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "to_numeric",
                        FunctionTransformer(convert_to_numeric, validate=False),
                    ),
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value=-999),
                    ),
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("label_pipeline", label_pipeline, label_cols),
                    ("frequency_pipeline", frequency_pipeline, freq_cols),
                    ("numeric_pipeline", numeric_pipeline, passthrough_cols),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )
        except Exception as error:
            raise CustomException(error, sys)  # type: ignore[arg-type]

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        try:
            logging.info("=" * 80)
            logging.info("DATA TRANSFORMATION STARTED")

            train_df = load_dataframe(train_path)
            test_df = load_dataframe(test_path)

            engineered_train_df = feature_engineering(
                train_df.drop(columns=["data_split"], errors="ignore"),
            )
            train_feature_assets = build_feature_engineering_assets(
                engineered_train_df.drop(columns=[TARGET_COLUMN], errors="ignore")
            )
            engineered_test_df = feature_engineering(
                test_df.drop(columns=["data_split"], errors="ignore"),
                feature_assets=train_feature_assets,
            )

            target_train = engineered_train_df[TARGET_COLUMN].astype(float)
            target_test = engineered_test_df[TARGET_COLUMN].astype(float)

            input_feature_train_df = engineered_train_df.drop(
                columns=[TARGET_COLUMN, ID_COLUMN, TIME_COLUMN],
                errors="ignore",
            )
            input_feature_test_df = engineered_test_df.drop(
                columns=[TARGET_COLUMN, ID_COLUMN, TIME_COLUMN],
                errors="ignore",
            )

            label_cols, freq_cols, passthrough_cols = validate_columns(
                input_feature_train_df,
                get_label_encode_cols(),
                get_freq_encode_cols(),
                get_passthrough_cols(),
            )

            preprocessor = self.get_data_transformer_object(
                label_cols,
                freq_cols,
                passthrough_cols,
            )

            train_features = preprocessor.fit_transform(input_feature_train_df)
            test_features = preprocessor.transform(input_feature_test_df)

            train_array = np.c_[train_features, target_train.to_numpy()]
            test_array = np.c_[test_features, target_test.to_numpy()]

            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            save_object(self.config.feature_assets_file_path, train_feature_assets)
            save_json(
                self.config.schema_file_path,
                {
                    "raw_input_columns": list(train_df.columns),
                    "engineered_input_columns": list(
                        extract_expected_model_columns(preprocessor)
                    ),
                    "label_columns": label_cols,
                    "frequency_columns": freq_cols,
                    "numeric_columns": passthrough_cols,
                },
            )

            logging.info(
                "Transformation complete. Train array: %s | Validation array: %s",
                train_array.shape,
                test_array.shape,
            )
            logging.info("=" * 80)

            return (
                train_array,
                test_array,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as error:
            raise CustomException(error, sys)  # type: ignore[arg-type]
