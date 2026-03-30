import os
import sys
import gc
from dataclasses import dataclass
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    load_dataframe,
    merge_dataframes,
    reduce_mem_usage,
    save_dataframe,
    time_based_split,
)


@dataclass
class DataIngestionConfig:
    train_transaction_path: str = os.path.join(
        "notebooks", "data", "train_transaction.csv"
    )
    train_identity_path: str = os.path.join(
        "notebooks", "data", "train_identity.csv"
    )
    test_transaction_path: str = os.path.join(
        "notebooks", "data", "test_transaction.csv"
    )
    test_identity_path: str = os.path.join(
        "notebooks", "data", "test_identity.csv"
    )
    train_data_path: str = os.path.join("artifacts", "train.parquet")
    val_data_path: str = os.path.join("artifacts", "val.parquet")
    test_data_path: str = os.path.join("artifacts", "test.parquet")
    raw_data_path: str = os.path.join("artifacts", "raw.parquet")


class DataIngestion:
    def __init__(self) -> None:
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str, str]:
        try:
            logging.info("=" * 80)
            logging.info("DATA INGESTION STARTED")

            train_transaction_df = load_dataframe(self.config.train_transaction_path)
            train_identity_df = load_dataframe(self.config.train_identity_path)
            train_transaction_df = reduce_mem_usage(train_transaction_df)
            train_identity_df = reduce_mem_usage(train_identity_df)

            train_df = merge_dataframes(
                train_transaction_df,
                train_identity_df,
                on=ID_COLUMN,
                how="left",
            )
            del train_transaction_df, train_identity_df
            gc.collect()

            train_df = reduce_mem_usage(train_df)
            train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype("float32")
            train_df[TIME_COLUMN] = train_df[TIME_COLUMN].astype("float32")
            train_df = pd.concat(
                [
                    train_df,
                    pd.DataFrame({"data_split": "train"}, index=train_df.index),
                ],
                axis=1,
            )

            save_dataframe(train_df, self.config.raw_data_path)

            train_split_df, validation_split_df = time_based_split(
                train_df,
                dt_col=TIME_COLUMN,
                ratio=0.8,
            )

            save_dataframe(train_split_df, self.config.train_data_path)
            save_dataframe(validation_split_df, self.config.val_data_path)

            logging.info(
                "Train ingestion complete. Train: %s | Validation: %s",
                train_split_df.shape,
                validation_split_df.shape,
            )

            del train_df, train_split_df, validation_split_df
            gc.collect()

            test_transaction_df = load_dataframe(self.config.test_transaction_path)
            test_identity_df = load_dataframe(self.config.test_identity_path)
            test_transaction_df = reduce_mem_usage(test_transaction_df)
            test_identity_df = reduce_mem_usage(test_identity_df)

            test_df = merge_dataframes(
                test_transaction_df,
                test_identity_df,
                on=ID_COLUMN,
                how="left",
            )
            del test_transaction_df, test_identity_df
            gc.collect()

            test_df = reduce_mem_usage(test_df)
            test_df = pd.concat(
                [
                    test_df,
                    pd.DataFrame({"data_split": "test"}, index=test_df.index),
                ],
                axis=1,
            )
            save_dataframe(test_df, self.config.test_data_path)

            logging.info(
                "Ingestion complete. Saved test dataset with shape: %s",
                test_df.shape,
            )  # type: ignore[arg-type]
            logging.info("=" * 80)

            return (
                self.config.train_data_path,
                self.config.val_data_path,
                self.config.test_data_path,
            )
        except Exception as error:
            raise CustomException(error, sys)  # type: ignore[arg-type]
