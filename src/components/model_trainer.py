import os
import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    evaluate_models,
    get_scale_pos_weight,
    save_confusion_matrix_image,
    save_json,
    save_object,
    save_roc_curve_image,
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_path: str = os.path.join("artifacts", "model_report.json")
    roc_curve_path: str = os.path.join("artifacts", "roc_curve.png")
    confusion_matrix_path: str = os.path.join("artifacts", "confusion_matrix.png")
    target_roc_auc: float = 0.90


class ModelTrainer:
    def __init__(self) -> None:
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray,
    ) -> tuple[float, str]:
        try:
            logging.info("=" * 80)
            logging.info("MODEL TRAINING STARTED")

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_val = test_arr[:, :-1]
            y_val = test_arr[:, -1]
            scale_pos_weight = get_scale_pos_weight(y_train)

            models = {
                "LightGBM": lgb.LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    learning_rate=0.03,
                    n_estimators=500,
                    num_leaves=255,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=50,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                "XGBoost": xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    learning_rate=0.05,
                    n_estimators=400,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "CatBoost": CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                    random_seed=42,
                    verbose=0,
                    thread_count=-1,
                ),
            }

            params = {
                "LightGBM": {},
                "XGBoost": {
                    "n_estimators": [300, 500],
                    "max_depth": [6, 8],
                    "learning_rate": [0.03, 0.05],
                },
                "CatBoost": {
                    "iterations": [300, 500],
                    "depth": [6, 8],
                    "learning_rate": [0.03, 0.05],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                models=models,
                params=params,
                cv=2,
            )

            best_model_name = max(
                model_report,
                key=lambda model_name: (
                    model_report[model_name]["roc_auc"],
                    model_report[model_name]["f1_score"],
                ),
            )
            best_result = model_report[best_model_name]
            best_model = best_result["best_model"]
            best_roc_auc = best_result["roc_auc"]

            logging.info("MODEL COMPARISON SUMMARY")
            for model_name, metrics in model_report.items():
                tag = " <- BEST" if model_name == best_model_name else ""
                logging.info(
                    "%s | ROC-AUC: %.6f | Accuracy: %.6f | F1: %.6f%s",
                    model_name,
                    metrics["roc_auc"],
                    metrics["accuracy"],
                    metrics["f1_score"],
                    tag,
                )

            save_object(self.config.trained_model_file_path, best_model)
            save_json(
                self.config.model_report_path,
                {
                    model_name: {
                        "roc_auc": round(metrics["roc_auc"], 6),
                        "accuracy": round(metrics["accuracy"], 6),
                        "f1_score": round(metrics["f1_score"], 6),
                    }
                    for model_name, metrics in model_report.items()
                }
                | {
                    "best_model": {
                        "name": best_model_name,
                        "roc_auc": round(best_result["roc_auc"], 6),
                        "accuracy": round(best_result["accuracy"], 6),
                        "f1_score": round(best_result["f1_score"], 6),
                    }
                },
            )

            save_roc_curve_image(
                y_val,
                best_result["y_prob"],
                self.config.roc_curve_path,
                best_model_name,
            )
            save_confusion_matrix_image(
                y_val,
                best_result["y_pred"],
                self.config.confusion_matrix_path,
                best_model_name,
            )

            if best_roc_auc < self.config.target_roc_auc:
                logging.warning(
                    "Best ROC-AUC %.4f is below target %.2f",
                    best_roc_auc,
                    self.config.target_roc_auc,
                )
            else:
                logging.info(
                    "Best ROC-AUC %.4f achieved by %s",
                    best_roc_auc,
                    best_model_name,
                )

            logging.info("MODEL TRAINING COMPLETED")
            logging.info("=" * 80)
            return best_roc_auc, self.config.trained_model_file_path
        except Exception as error:
            raise CustomException(error, sys)  # type: ignore[arg-type]
