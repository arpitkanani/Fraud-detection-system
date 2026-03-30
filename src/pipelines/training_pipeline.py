import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_training_pipeline() -> tuple[float, str]:
    try:
        logging.info("TRAINING PIPELINE EXECUTION STARTED")
        data_ingestion = DataIngestion()
        train_path, validation_path, _ = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_array, validation_array, _ = data_transformation.initiate_data_transformation(
            train_path,
            validation_path,
        )

        model_trainer = ModelTrainer()
        best_roc_auc, model_path = model_trainer.initiate_model_trainer(
            train_array,
            validation_array,
        )
        logging.info("TRAINING PIPELINE EXECUTION FINISHED")
        return best_roc_auc, model_path
    except Exception as error:
        raise CustomException(error, sys)  # type: ignore[arg-type]


if __name__ == "__main__":
    roc_auc, trained_model_path = run_training_pipeline()
    print(f"Best ROC-AUC: {roc_auc:.4f}")
    print(f"Model saved to: {trained_model_path}")
