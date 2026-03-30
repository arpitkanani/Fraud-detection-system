import json
import os
import pickle
import sys
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


matplotlib.use("Agg")


TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"

TRANSACTION_BASE_COLUMNS = [
    ID_COLUMN,
    TIME_COLUMN,
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
]
TRANSACTION_BASE_COLUMNS += [f"C{i}" for i in range(1, 15)]
TRANSACTION_BASE_COLUMNS += ["D1", "D2", "D4", "D10", "D11"]
TRANSACTION_BASE_COLUMNS += [f"M{i}" for i in range(1, 10)]

IDENTITY_BASE_COLUMNS = [
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_09",
    "id_10",
    "id_11",
    "id_12",
    "id_13",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]

V_BASE_COLUMNS = [f"V{i}" for i in range(1, 138)]

DEFAULT_RAW_INPUT_COLUMNS = (
    TRANSACTION_BASE_COLUMNS + IDENTITY_BASE_COLUMNS + V_BASE_COLUMNS
)


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        deduped_df = df.copy()
        duplicate_names = deduped_df.columns[deduped_df.columns.duplicated()].unique()
        for column_name in duplicate_names:
            duplicate_slice = deduped_df.loc[:, deduped_df.columns == column_name]
            deduped_df[column_name] = duplicate_slice.bfill(axis=1).iloc[:, 0]
        deduped_df = deduped_df.loc[:, ~deduped_df.columns.duplicated()]
        return deduped_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    try:
        normalized_df = df.copy()
        normalized_df.columns = [
            str(column_name).strip().replace("-", "_")
            for column_name in normalized_df.columns
        ]
        normalized_df = deduplicate_columns(normalized_df)
        return normalized_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def load_dataframe(source: str | bytes) -> pd.DataFrame:
    try:
        logging.info("Loading tabular data with PyArrow")
        read_options = pa_csv.ReadOptions(use_threads=True)
        parse_options = pa_csv.ParseOptions(delimiter=",")
        convert_options = pa_csv.ConvertOptions(strings_can_be_null=True)

        if isinstance(source, bytes):
            table = pa_csv.read_csv(
                pa.BufferReader(source),
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            )
        else:
            if source.lower().endswith(".parquet"): # type: ignore
                table = pa_parquet.read_table(source)
            else:
                table = pa_csv.read_csv(
                    source,
                    read_options=read_options,
                    parse_options=parse_options,
                    convert_options=convert_options,
                )

        dataframe = table.to_pandas()
        dataframe = normalize_column_names(dataframe)
        logging.info("Loaded dataframe shape: %s", dataframe.shape)
        return dataframe
    except Exception as error:
        if "received signal 2" in str(error).lower():
            error = RuntimeError(
                "CSV loading was interrupted while the server was processing the upload. "
                "Run the app without auto-reload and upload the file again."
            )
        raise CustomException(error, sys)  # type: ignore[arg-type]


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if file_path.lower().endswith(".parquet"):
            table = pa.Table.from_pandas(df, preserve_index=False)
            pa_parquet.write_table(table, file_path)
        elif file_path.lower().endswith(".csv"):
            df.to_csv(file_path, index=False)
        elif file_path.lower().endswith(".json"):
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(df, file, indent=4)
        else:
            raise ValueError(f"Unsupported output format for path: {file_path}")
        logging.info("Saved dataframe artifact to %s", file_path)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def save_json(file_path: str, payload: dict | list) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        logging.info("Saved json artifact to %s", file_path)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def load_json(file_path: str) -> dict | list:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def save_object(file_path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info("Saved object to %s", file_path)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file:
            loaded_object = pickle.load(file)
        logging.info("Loaded object from %s", file_path)
        return loaded_object
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str = ID_COLUMN,
    how: str = "left",
) -> pd.DataFrame:
    try:
        left_normalized = normalize_column_names(left_df)
        right_normalized = normalize_column_names(right_df)
        merged_df = left_normalized.merge(right_normalized, on=on, how=how) # type: ignore
        logging.info(
            "Merged dataframes on %s with %s join. Shape: %s",
            on,
            how,
            merged_df.shape,
        )
        return merged_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def time_based_split(
    df: pd.DataFrame,
    dt_col: str = TIME_COLUMN,
    ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if dt_col in df.columns:
            dt_values = pd.to_numeric(df[dt_col], errors="coerce").to_numpy(copy=True)
            nan_mask = np.isnan(dt_values)
            if nan_mask.any():
                if (~nan_mask).any():
                    dt_values[nan_mask] = np.nanmax(dt_values[~nan_mask]) + 1
                else:
                    dt_values[nan_mask] = 0
            sorted_positions = np.argsort(dt_values, kind="mergesort")
        else:
            sorted_positions = np.arange(len(df))

        split_index = int(len(sorted_positions) * ratio)
        train_positions = sorted_positions[:split_index]
        validation_positions = sorted_positions[split_index:]

        train_df = df.take(train_positions)
        validation_df = df.take(validation_positions)
        logging.info(
            "Time-based split complete. Train: %s | Validation: %s",
            train_df.shape,
            validation_df.shape,
        )
        return train_df, validation_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    try:
        optimized_df = df
        start_memory_mb = optimized_df.memory_usage(deep=True).sum() / (1024 ** 2)

        for column_name in optimized_df.columns:
            column = optimized_df[column_name]
            if pd.api.types.is_integer_dtype(column):
                optimized_df[column_name] = pd.to_numeric(column, downcast="integer")
            elif pd.api.types.is_float_dtype(column):
                optimized_df[column_name] = pd.to_numeric(column, downcast="float")

        end_memory_mb = optimized_df.memory_usage(deep=True).sum() / (1024 ** 2)
        logging.info(
            "Memory optimized from %.2f MB to %.2f MB",
            start_memory_mb,
            end_memory_mb,
        )
        return optimized_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def align_dataframe_to_columns(
    df: pd.DataFrame,
    expected_columns: Iterable[str],
    fill_value: Any = np.nan,
) -> pd.DataFrame:
    try:
        aligned_df = normalize_column_names(df)
        expected_column_list = list(expected_columns)
        missing_columns = [
            column_name
            for column_name in expected_column_list
            if column_name not in aligned_df.columns
        ]
        if missing_columns:
            missing_df = pd.DataFrame(
                fill_value,
                index=aligned_df.index,
                columns=missing_columns,
            )
            aligned_df = pd.concat([aligned_df, missing_df], axis=1)
        return aligned_df.loc[:, expected_column_list]
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def get_label_encode_cols() -> list[str]:
    return [
        "ProductCD",
        "card4",
        "card6",
        "DeviceType",
        "id_12",
        "id_15",
        "id_16",
        "id_28",
        "id_29",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
    ]


def get_freq_encode_cols() -> list[str]:
    return [
        "card1",
        "P_emaildomain",
        "R_emaildomain",
        "id_30",
        "id_31",
        "id_33",
        "DeviceInfo",
        "card1_addr1",
        "card1_card2",
        "card4_card6",
        "card1_ProductCD",
        "addr1_addr2",
        "P_R_email_match_domain",
    ]


def get_passthrough_cols() -> list[str]:
    return [
        "TransactionAmt",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "D1",
        "D2",
        "D4",
        "D10",
        "D11",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "id_01",
        "id_02",
        "id_05",
        "id_06",
        "id_09",
        "id_10",
        "id_11",
        "id_13",
        "TransactionAmt_Log",
        "TransactionAmt_decimal",
        "TransactionAmt_is_round",
        "TransactionAmt_bin",
        "TransactionAmt_cents",
        "TransactionAmt_ends_00",
        "TransactionAmt_ends_99",
        "TransactionAmt_ends_95",
        "Transaction_hour",
        "Transaction_dow",
        "Transaction_day",
        "Transaction_week",
        "Transaction_is_weekend",
        "Transaction_is_business_hour",
        "addr1_missing",
        "addr2_missing",
        "both_addr_missing",
        "dist1_log",
        "dist1_missing",
        "v1_sum",
        "v1_mean",
        "v1_std",
        "v1_nan_count",
        "v2_sum",
        "v2_mean",
        "v2_std",
        "v2_nan_count",
        "v3_sum",
        "v3_mean",
        "v3_std",
        "v3_nan_count",
        "v4_sum",
        "v4_mean",
        "v4_std",
        "v4_nan_count",
        "v5_sum",
        "v5_mean",
        "v5_std",
        "v5_nan_count",
        "v6_sum",
        "v6_mean",
        "v6_std",
        "v6_nan_count",
        "v7_sum",
        "v7_mean",
        "v7_std",
        "v7_nan_count",
        "V_sum_all",
        "V_mean_all",
        "V_std_all",
        "V_nan_count_all",
        "V_nan_ratio",
        "id_num_nan_count",
        "id_num_mean",
        "id_num_std",
        "id_12_isFound",
        "id_15_isNew",
        "id_15_isFound",
        "id_16_isFound",
        "id_28_isNew",
        "id_28_isFound",
        "id_29_isFound",
        "id_34_match",
        "id_36_isT",
        "id_37_isT",
        "id_38_isT",
        "card1_freq",
        "card1_addr1_freq",
        "P_emaildomain_freq",
        "R_emaildomain_freq",
        "card1_TransactionAmt_mean",
        "card1_TransactionAmt_std",
        "client_uid_freq",
        "email_match",
        "device_is_mobile",
        "device_is_desktop",
        "screen_width",
        "screen_height",
        "screen_area",
    ]


def validate_columns(
    df: pd.DataFrame,
    label_cols: list[str],
    freq_cols: list[str],
    passthrough_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    try:
        label_columns = [column for column in label_cols if column in df.columns]
        frequency_columns = [column for column in freq_cols if column in df.columns]
        passthrough_columns = [
            column for column in passthrough_cols if column in df.columns
        ]
        logging.info(
            "Validated transformer columns. Label: %s | Frequency: %s | Numeric: %s",
            len(label_columns),
            len(frequency_columns),
            len(passthrough_columns),
        )
        return label_columns, frequency_columns, passthrough_columns
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def convert_to_string_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    try:
        return pd.DataFrame(X).astype("string").fillna("missing").astype(str)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def convert_to_numeric_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    try:
        dataframe = pd.DataFrame(X).copy()
        for column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")
        return dataframe
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def coerce_columns_to_numeric(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> pd.DataFrame:
    try:
        coerced_df = df.copy()
        for column_name in numeric_columns:
            if column_name in coerced_df.columns:
                coerced_df[column_name] = pd.to_numeric(
                    coerced_df[column_name],
                    errors="coerce",
                )
        return coerced_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.mappings_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "FrequencyEncoder":
        dataframe = pd.DataFrame(X)
        self.mappings_ = []
        for column_name in dataframe.columns:
            frequencies = (
                dataframe[column_name]
                .astype("string")
                .fillna("missing")
                .value_counts(normalize=True)
                .to_dict()
            )
            self.mappings_.append(frequencies) # type: ignore
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        dataframe = pd.DataFrame(X)
        encoded_columns: list[np.ndarray] = []
        for column_index, column_name in enumerate(dataframe.columns):
            mapping = self.mappings_[column_index]
            encoded_column = (
                dataframe[column_name]
                .astype("string")
                .fillna("missing")
                .map(mapping)
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )
            encoded_columns.append(encoded_column)
        return np.column_stack(encoded_columns)


def frequency_encoder(X: np.ndarray) -> np.ndarray:
    dataframe = pd.DataFrame(X)
    encoded_columns: list[np.ndarray] = []
    for column_name in dataframe.columns:
        frequencies = (
            dataframe[column_name]
            .astype("string")
            .fillna("missing")
            .value_counts(normalize=True)
            .to_dict()
        )
        encoded_column = (
            dataframe[column_name]
            .astype("string")
            .fillna("missing")
            .map(frequencies)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        encoded_columns.append(encoded_column)
    return np.column_stack(encoded_columns)


def _numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column_name], errors="coerce")


def _string_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series("missing", index=df.index, dtype="object")
    return (
        df[column_name]
        .replace("", np.nan)
        .astype("string")
        .fillna("missing")
        .astype(str)
    )


def fe_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    try:
        transaction_amount = _numeric_series(df, "TransactionAmt").fillna(0.0)
        amount_integer = np.floor(transaction_amount).astype(np.int64)
        cents = ((transaction_amount.fillna(0.0) * 100).round() % 100).astype(np.int64)

        new_columns = {
            "TransactionAmt_Log": np.log1p(transaction_amount.clip(lower=0.0)),
            "TransactionAmt_decimal": (
                ((transaction_amount - amount_integer) * 1000).round().astype(np.int64)
            ),
            "TransactionAmt_is_round": (transaction_amount == amount_integer).astype(int),
            "TransactionAmt_bin": pd.cut(
                transaction_amount,
                bins=[0, 50, 100, 200, 500, 1000, 5000, 10000, np.inf],
                labels=False,
                include_lowest=True,
            ),
            "TransactionAmt_cents": cents,
            "TransactionAmt_ends_00": (cents == 0).astype(int),
            "TransactionAmt_ends_99": (cents == 99).astype(int),
            "TransactionAmt_ends_95": (cents == 95).astype(int),
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_time_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        transaction_dt = _numeric_series(df, TIME_COLUMN).fillna(0)
        transaction_hour = ((transaction_dt // 3600) % 24).astype(int)
        transaction_dow = ((transaction_dt // 86400) % 7).astype(int)

        new_columns = {
            "Transaction_hour": transaction_hour,
            "Transaction_dow": transaction_dow,
            "Transaction_day": (transaction_dt // 86400).astype(int),
            "Transaction_week": (transaction_dt // 604800).astype(int),
            "Transaction_is_weekend": (transaction_dow >= 5).astype(int),
            "Transaction_is_business_hour": (
                (transaction_hour >= 9) & (transaction_hour <= 17)
            ).astype(int),
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_card_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        new_columns = {
            "card1_card2": _string_series(df, "card1") + "_" + _string_series(df, "card2"),
            "card4_card6": _string_series(df, "card4") + "_" + _string_series(df, "card6"),
            "card1_addr1": _string_series(df, "card1") + "_" + _string_series(df, "addr1"),
            "card1_ProductCD": _string_series(df, "card1") + "_" + _string_series(df, "ProductCD"),
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_email_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        purchaser_domain = _string_series(df, "P_emaildomain")
        receiver_domain = _string_series(df, "R_emaildomain")

        new_columns = {
            "email_match": (purchaser_domain == receiver_domain).astype(int),
            "P_R_email_match_domain": purchaser_domain + "_" + receiver_domain,
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_device_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        device_type = _string_series(df, "DeviceType").str.lower()
        screen_resolution = _string_series(df, "id_33")
        split_resolution = screen_resolution.str.extract(r"(?P<width>\d+)\s*x\s*(?P<height>\d+)")
        screen_width = pd.to_numeric(split_resolution["width"], errors="coerce")
        screen_height = pd.to_numeric(split_resolution["height"], errors="coerce")

        new_columns = {
            "device_is_mobile": (device_type == "mobile").astype(int),
            "device_is_desktop": (device_type == "desktop").astype(int),
            "screen_width": screen_width,
            "screen_height": screen_height,
            "screen_area": screen_width * screen_height,
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_address_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        addr1 = _numeric_series(df, "addr1")
        addr2 = _numeric_series(df, "addr2")
        dist1 = _numeric_series(df, "dist1")

        new_columns = {
            "addr1_missing": addr1.isna().astype(int),
            "addr2_missing": addr2.isna().astype(int),
            "both_addr_missing": (addr1.isna() & addr2.isna()).astype(int),
            "addr1_addr2": _string_series(df, "addr1") + "_" + _string_series(df, "addr2"),
            "dist1_missing": dist1.isna().astype(int),
            "dist1_log": np.log1p(dist1.fillna(0.0).clip(lower=0.0)),
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_v_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    try:
        numeric_v_frame = df[V_BASE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        v_groups = {
            "v1": [f"V{i}" for i in range(1, 12)],
            "v2": [f"V{i}" for i in range(12, 27)],
            "v3": [f"V{i}" for i in range(27, 35)],
            "v4": [f"V{i}" for i in range(35, 53)],
            "v5": [f"V{i}" for i in range(53, 75)],
            "v6": [f"V{i}" for i in range(75, 95)],
            "v7": [f"V{i}" for i in range(95, 138)],
        }

        new_columns: dict[str, pd.Series] = {}
        for group_name, group_columns in v_groups.items():
            group_frame = numeric_v_frame[group_columns]
            new_columns[f"{group_name}_sum"] = group_frame.sum(axis=1)
            new_columns[f"{group_name}_mean"] = group_frame.mean(axis=1)
            new_columns[f"{group_name}_std"] = group_frame.std(axis=1)
            new_columns[f"{group_name}_nan_count"] = group_frame.isna().sum(axis=1)

        new_columns["V_sum_all"] = numeric_v_frame.sum(axis=1)
        new_columns["V_mean_all"] = numeric_v_frame.mean(axis=1)
        new_columns["V_std_all"] = numeric_v_frame.std(axis=1)
        new_columns["V_nan_count_all"] = numeric_v_frame.isna().sum(axis=1)
        new_columns["V_nan_ratio"] = (
            new_columns["V_nan_count_all"] / max(len(V_BASE_COLUMNS), 1)
        )

        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def fe_id_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        numeric_id_columns = [
            "id_01",
            "id_02",
            "id_05",
            "id_06",
            "id_09",
            "id_10",
            "id_11",
            "id_13",
        ]
        numeric_id_frame = df[numeric_id_columns].apply(pd.to_numeric, errors="coerce")
        id_34 = _string_series(df, "id_34")

        new_columns = {
            "id_num_nan_count": numeric_id_frame.isna().sum(axis=1),
            "id_num_mean": numeric_id_frame.mean(axis=1),
            "id_num_std": numeric_id_frame.std(axis=1),
            "id_12_isFound": (_string_series(df, "id_12") == "Found").astype(int),
            "id_15_isNew": (_string_series(df, "id_15") == "New").astype(int),
            "id_15_isFound": (_string_series(df, "id_15") == "Found").astype(int),
            "id_16_isFound": (_string_series(df, "id_16") == "Found").astype(int),
            "id_28_isNew": (_string_series(df, "id_28") == "New").astype(int),
            "id_28_isFound": (_string_series(df, "id_28") == "Found").astype(int),
            "id_29_isFound": (_string_series(df, "id_29") == "Found").astype(int),
            "id_34_match": pd.to_numeric(
                id_34.str.extract(r":\s*(?P<match>\d+)")["match"],
                errors="coerce",
            ).fillna(-1),
            "id_36_isT": (_string_series(df, "id_36") == "T").astype(int),
            "id_37_isT": (_string_series(df, "id_37") == "T").astype(int),
            "id_38_isT": (_string_series(df, "id_38") == "T").astype(int),
        }
        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def build_feature_engineering_assets(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    try:
        assets = {
            "card1_freq": (
                _string_series(df, "card1").value_counts(normalize=True).to_dict()
            ),
            "card1_addr1_freq": (
                _string_series(df, "card1_addr1").value_counts(normalize=True).to_dict()
            ),
            "P_emaildomain_freq": (
                _string_series(df, "P_emaildomain").value_counts(normalize=True).to_dict()
            ),
            "R_emaildomain_freq": (
                _string_series(df, "R_emaildomain").value_counts(normalize=True).to_dict()
            ),
            "card1_TransactionAmt_mean": (
                df.assign(
                    card1_key=_string_series(df, "card1"),
                    transaction_amount=_numeric_series(df, "TransactionAmt"),
                )
                .groupby("card1_key")["transaction_amount"]
                .mean()
                .to_dict()
            ),
            "card1_TransactionAmt_std": (
                df.assign(
                    card1_key=_string_series(df, "card1"),
                    transaction_amount=_numeric_series(df, "TransactionAmt"),
                )
                .groupby("card1_key")["transaction_amount"]
                .std()
                .to_dict()
            ),
        }

        client_uid_series = (
            _string_series(df, "card1")
            + "_"
            + _string_series(df, "addr1")
            + "_"
            + _string_series(df, "D1")
        )
        assets["client_uid_freq"] = client_uid_series.value_counts(
            normalize=True
        ).to_dict()

        return assets
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def apply_feature_assets(
    df: pd.DataFrame,
    feature_assets: dict[str, dict[str, float]],
) -> pd.DataFrame:
    try:
        card1 = _string_series(df, "card1")
        card1_addr1 = _string_series(df, "card1_addr1")
        purchaser_domain = _string_series(df, "P_emaildomain")
        receiver_domain = _string_series(df, "R_emaildomain")
        client_uid = card1 + "_" + _string_series(df, "addr1") + "_" + _string_series(df, "D1")

        new_columns = {
            "card1_freq": card1.map(feature_assets.get("card1_freq", {})).fillna(0.0),
            "card1_addr1_freq": card1_addr1.map(
                feature_assets.get("card1_addr1_freq", {})
            ).fillna(0.0),
            "P_emaildomain_freq": purchaser_domain.map(
                feature_assets.get("P_emaildomain_freq", {})
            ).fillna(0.0),
            "R_emaildomain_freq": receiver_domain.map(
                feature_assets.get("R_emaildomain_freq", {})
            ).fillna(0.0),
            "card1_TransactionAmt_mean": card1.map(
                feature_assets.get("card1_TransactionAmt_mean", {})
            ).fillna(-999.0),
            "card1_TransactionAmt_std": card1.map(
                feature_assets.get("card1_TransactionAmt_std", {})
            ).fillna(-999.0),
            "client_uid_freq": client_uid.map(
                feature_assets.get("client_uid_freq", {})
            ).fillna(0.0),
        }

        return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def feature_engineering(
    df: pd.DataFrame,
    feature_assets: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    try:
        logging.info("Running feature engineering")
        target_series = None
        if TARGET_COLUMN in df.columns:
            target_series = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

        working_df = align_dataframe_to_columns(
            df.drop(columns=[TARGET_COLUMN], errors="ignore"),
            DEFAULT_RAW_INPUT_COLUMNS,
            np.nan,
        )
        working_df = reduce_mem_usage(working_df)
        working_df = fe_transaction_amount(working_df)
        working_df = fe_time_features(working_df)
        working_df = fe_card_features(working_df)
        working_df = fe_email_features(working_df)
        working_df = fe_device_features(working_df)
        working_df = fe_address_features(working_df)
        working_df = fe_v_aggregations(working_df)
        working_df = fe_id_features(working_df)

        assets = feature_assets or build_feature_engineering_assets(working_df)
        working_df = apply_feature_assets(working_df, assets)
        if target_series is not None:
            working_df[TARGET_COLUMN] = target_series.to_numpy()
        working_df = reduce_mem_usage(working_df)
        logging.info("Feature engineering complete. Shape: %s", working_df.shape)
        return working_df
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def get_scale_pos_weight(y: np.ndarray) -> float:
    try:
        y_array = np.asarray(y).astype(float)
        y_array = y_array[~np.isnan(y_array)]
        negative_count = np.sum(y_array == 0)
        positive_count = np.sum(y_array == 1)
        if positive_count == 0:
            return 1.0
        scale_pos_weight = float(negative_count) / float(positive_count)
        logging.info("scale_pos_weight computed as %.4f", scale_pos_weight)
        return scale_pos_weight
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: Any,
) -> dict[str, Any]:
    try:
        model.fit(X_train, y_train)
        y_probability = model.predict_proba(X_val)[:, 1]
        y_prediction = model.predict(X_val)

        roc_auc = roc_auc_score(y_val, y_probability)
        accuracy = accuracy_score(y_val, y_prediction)
        f1 = f1_score(y_val, y_prediction, zero_division=0)
        report = classification_report(y_val, y_prediction, zero_division=0)

        logging.info("Validation ROC-AUC: %.6f", roc_auc)
        logging.info("Validation Accuracy: %.6f", accuracy)
        logging.info("Validation F1-score: %.6f", f1)
        logging.info("\n%s", report)

        return {
            "model": model,
            "roc_auc": float(roc_auc),
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "y_prob": y_probability,
            "y_pred": y_prediction,
            "classification_report": report,
        }
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    models: dict[str, Any],
    params: dict[str, dict[str, list[Any]]],
    cv: int = 2,
) -> dict[str, dict[str, Any]]:
    try:
        evaluation_report: dict[str, dict[str, Any]] = {}

        for model_name, model in models.items():
            logging.info("Training candidate model: %s", model_name)
            model_params = params.get(model_name, {})

            if model_params:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=1,
                    verbose=0,
                )
                grid_search.fit(X_train, y_train)
                fitted_model = grid_search.best_estimator_
                logging.info(
                    "%s best params: %s",
                    model_name,
                    grid_search.best_params_,
                )
            else:
                fitted_model = model
                fitted_model.fit(X_train, y_train)
                logging.info("%s used base parameters without tuning", model_name)

            y_probability = fitted_model.predict_proba(X_val)[:, 1]
            y_prediction = fitted_model.predict(X_val)
            roc_auc = roc_auc_score(y_val, y_probability)
            accuracy = accuracy_score(y_val, y_prediction)
            f1 = f1_score(y_val, y_prediction, zero_division=0)

            evaluation_report[model_name] = {
                "best_model": fitted_model,
                "roc_auc": float(roc_auc),
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "y_prob": y_probability,
                "y_pred": y_prediction,
            }

            logging.info(
                "%s validation ROC-AUC: %.6f | Accuracy: %.6f | F1: %.6f",
                model_name,
                roc_auc,
                accuracy,
                f1,
            )
            logging.info(
                "\n%s",
                classification_report(y_val, y_prediction, zero_division=0),
            )

        return evaluation_report
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def save_roc_curve_image(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    model_name: str,
) -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={auc_value:.4f})",
            color="#15803d",
            linewidth=2,
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Fraud Detection ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info("Saved ROC curve to %s", save_path)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def save_confusion_matrix_image(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    model_name: str,
) -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(6, 5))
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=["Legit", "Fraud"],
        )
        display.plot(ax=axis, cmap="Greens", colorbar=False)
        axis.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(figure)
        logging.info("Saved confusion matrix to %s", save_path)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def extract_expected_model_columns(preprocessor: Any) -> list[str]:
    try:
        if hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

        expected_columns: list[str] = []
        if hasattr(preprocessor, "transformers_"):
            for _, _, columns in preprocessor.transformers_:
                if isinstance(columns, (list, tuple, pd.Index, np.ndarray)):
                    expected_columns.extend(list(columns))

        return list(dict.fromkeys(expected_columns))
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


def get_default_prediction_input_columns() -> list[str]:
    return DEFAULT_RAW_INPUT_COLUMNS.copy()


def map_risk_factor(probability: float) -> str:
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "NO_RISK"


def save_prediction_preview_image(results: pd.DataFrame, save_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        preview = results["RiskFactor"].value_counts().reindex(
            ["HIGH", "MEDIUM", "NO_RISK"],
            fill_value=0,
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.barplot(
            x=preview.index,
            y=preview.values,
            hue=preview.index,
            palette=["#dc2626", "#f59e0b", "#16a34a"],
            ax=axis,
            legend=False,
        )
        axis.set_title("Prediction Risk Distribution")
        axis.set_xlabel("Risk Factor")
        axis.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(figure)
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]
