import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import json
import boto3
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
logger = logging.getLogger(__name__)


class CryptoLightGBMModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.model_config = config["model"]
        self.feature_importance = None

    def create_lgb_datasets(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[lgb.Dataset, lgb.Dataset]:
        """Create LightGBM datasets"""
        train_data = lgb.Dataset(
            X_train, label=y_train, feature_name=list(X_train.columns)
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, reference=train_data, feature_name=list(X_val.columns)
        )
        return train_data, val_data

    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Optuna objective function for hyperparameter optimization"""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbose": -1,
        }

        train_data, val_data = self.create_lgb_datasets(X_train, y_train, X_val, y_val)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 100,
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
        )

        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")

        return study.best_params

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        symbol: str,
        optimize_params: bool = True,
    ) -> Dict:
        """Train the LightGBM model with MLflow tracking"""

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"lightgbm_training_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):

            # Optimize hyperparameters if requested
            if optimize_params:
                best_params = self.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val
                )
                # Update model config with optimized params
                model_params = {**self.model_config, **best_params}
            else:
                model_params = self.model_config.copy()

            # Log parameters
            mlflow.log_params(
                {
                    "symbol": symbol,
                    **model_params,
                    "n_features": X_train.shape[1],
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "optimize_params": optimize_params,
                }
            )

            # Create datasets
            train_data, val_data = self.create_lgb_datasets(
                X_train, y_train, X_val, y_val
            )

            # Train model
            self.model = lgb.train(
                model_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=["train", "val"],
                num_boost_round=model_params.get("num_boost_round", 1000),
                callbacks=[
                    lgb.early_stopping(model_params.get("early_stopping_rounds", 100)),
                    lgb.log_evaluation(100),
                ],
            )

            # Make predictions
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "best_iteration": self.model.best_iteration,
                }
            )

            # Feature importance
            self.feature_importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": self.model.feature_importance(importance_type="gain"),
                }
            ).sort_values("importance", ascending=False)

            # Log feature importance
            mlflow.log_text(
                self.feature_importance.to_string(), "feature_importance.txt"
            )

            # Log model
            mlflow.lightgbm.log_model(
                self.model,
                "model",
                registered_model_name=f"{self.config['mlflow']['model_name']}_{symbol.replace('-', '_')}",
            )

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Training completed. MLflow run ID: {run_id}")
            
            # Save metrics to S3
            self.save_metrics_to_s3(
                {
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "best_iteration": self.model.best_iteration,
                },
                symbol
            )

            # Sauvegarde référence datasets
            self.save_reference_data_to_s3(X_train, y_train, X_val, y_val, symbol, run_id)
            
            # Sauvegarde des prédictions de référence
            self.save_reference_predictions_to_s3(X_train, y_train, train_pred, X_val, y_val, val_pred, symbol, run_id)

            return {
                "model": self.model,
                "metrics": {
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                },
                "feature_importance": self.feature_importance,
                "run_id": run_id,
                "best_params": model_params if optimize_params else None,
            }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        predictions = self.model.predict(X)
        return predictions

    def save_reference_data_to_s3(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        symbol: str,
        run_id: str,
    ):
        """
        Save training and validation datasets to S3 as CSV.
        """
        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.config["aws"]["region"],
                aws_access_key_id=self.config["aws"]["access_key_id"],
                aws_secret_access_key=self.config["aws"]["secret_access_key"],
                endpoint_url=self.config["aws"]["endpoint_url"],
            )
            bucket = self.config["aws"]["s3_bucket"]

            # Merge features + target
            train_df = X_train.copy()
            train_df["target"] = y_train
            val_df = X_val.copy()
            val_df["target"] = y_val

            # Save local
            train_path = f"/tmp/train_{symbol}_{run_id}.csv"
            val_path = f"/tmp/val_{symbol}_{run_id}.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

            # S3 keys
            s3_train_key = f"{self.config['aws']['s3_model_prefix']}/reference/train_{run_id}.csv"
            s3_val_key = f"{self.config['aws']['s3_model_prefix']}/reference/val_{run_id}.csv"

            # Upload to S3
            with open(train_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket, Key=s3_train_key, Body=f.read(), ContentType="text/csv"
                )
            with open(val_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket, Key=s3_val_key, Body=f.read(), ContentType="text/csv"
                )

            # Clean local files
            os.remove(train_path)
            os.remove(val_path)

            logger.info(f"Saved training reference data to S3: s3://{bucket}/{s3_train_key}")
            logger.info(f"Saved validation reference data to S3: s3://{bucket}/{s3_val_key}")

        except Exception as e:
            logger.error(f"Error saving reference data to S3: {str(e)}")

    def save_reference_predictions_to_s3(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_pred: np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        val_pred: np.ndarray,
        symbol: str,
        run_id: str,
    ):
        """
        Save training and validation predictions to S3 as CSV for Evidently monitoring.
        """
        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.config["aws"]["region"],
                aws_access_key_id=self.config["aws"]["access_key_id"],
                aws_secret_access_key=self.config["aws"]["secret_access_key"],
                endpoint_url=self.config["aws"]["endpoint_url"],
            )
            bucket = self.config["aws"]["s3_bucket"]

            # Create dataframes with target and prediction columns
            train_pred_df = pd.DataFrame({
                "target": y_train,
                "prediction": train_pred
            })
            
            val_pred_df = pd.DataFrame({
                "target": y_val,
                "prediction": val_pred
            })

            # Save local
            train_pred_path = f"/tmp/train_predictions_{symbol}_{run_id}.csv"
            val_pred_path = f"/tmp/val_predictions_{symbol}_{run_id}.csv"
            train_pred_df.to_csv(train_pred_path, index=False)
            val_pred_df.to_csv(val_pred_path, index=False)

            # S3 keys
            s3_train_pred_key = f"{self.config['aws']['s3_model_prefix']}/reference/train_predictions_{run_id}.csv"
            s3_val_pred_key = f"{self.config['aws']['s3_model_prefix']}/reference/val_predictions_{run_id}.csv"

            # Upload to S3
            with open(train_pred_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket, Key=s3_train_pred_key, Body=f.read(), ContentType="text/csv"
                )
            with open(val_pred_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket, Key=s3_val_pred_key, Body=f.read(), ContentType="text/csv"
                )

            # Clean local files
            os.remove(train_pred_path)
            os.remove(val_pred_path)

            logger.info(f"Saved training predictions to S3: s3://{bucket}/{s3_train_pred_key}")
            logger.info(f"Saved validation predictions to S3: s3://{bucket}/{s3_val_pred_key}")

        except Exception as e:
            logger.error(f"Error saving reference predictions to S3: {str(e)}")

    def save_metrics_to_s3(self, metrics: Dict, symbol: str):
        """
        Save training metrics to S3 as JSON.
        """
        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.config["aws"]["region"],
                aws_access_key_id=self.config["aws"]["access_key_id"],
                aws_secret_access_key=self.config["aws"]["secret_access_key"],
                endpoint_url=self.config["aws"]["endpoint_url"],
            )
            bucket = self.config["aws"]["s3_bucket"]

            metrics_path = f"/tmp/metrics_{symbol}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            s3_key = (
                f"{self.config['aws']['s3_model_prefix']}/metrics/metrics.json"
            )

            with open(metrics_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=f.read(),
                    ContentType="application/json",
                )

            os.remove(metrics_path)

            logger.info(f"Saved training metrics to S3: s3://{bucket}/{s3_key}")

        except Exception as e:
            logger.error(f"Error saving training metrics to S3: {str(e)}")

    def save_model_to_s3(self, symbol: str, run_id: str):

        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.config["aws"]["region"],
                aws_access_key_id=self.config["aws"]["access_key_id"],
                aws_secret_access_key=self.config["aws"]["secret_access_key"],
                endpoint_url=self.config["aws"]["endpoint_url"],
            )
            bucket = self.config["aws"]["s3_bucket"]

            # Save model
            model_path = f"/tmp/model_{symbol}_{run_id}.txt"
            self.model.save_model(model_path)

            s3_key = (
                f"{self.config['aws']['s3_model_prefix']}{symbol}/model_{run_id}.txt"
            )

            with open(model_path, "rb") as f:
                s3_client.put_object(
                    Bucket=bucket, Key=s3_key, Body=f.read(), ContentType="text/plain"
                )

            # Save feature importance
            if self.feature_importance is not None:
                importance_path = f"/tmp/feature_importance_{symbol}_{run_id}.csv"
                self.feature_importance.to_csv(importance_path, index=False)

                importance_s3_key = f"{self.config['aws']['s3_model_prefix']}{symbol}/feature_importance_{run_id}.csv"
                with open(importance_path, "rb") as f:
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=importance_s3_key,
                        Body=f.read(),
                        ContentType="text/csv",
                    )

                os.remove(importance_path)

            logger.info(f"Saved model to S3: s3://{bucket}/{s3_key}")
            os.remove(model_path)

        except Exception as e:
            logger.error(f"Error saving model to S3: {str(e)}")

    def load_model_from_s3(self, symbol: str, run_id: str):

        try:
            s3_client = boto3.client(
                "s3",
                region_name=self.config["aws"]["region"],
                aws_access_key_id=self.config["aws"]["access_key_id"],
                aws_secret_access_key=self.config["aws"]["secret_access_key"],
                endpoint_url=self.config["aws"]["endpoint_url"],
            )
            bucket = self.config["aws"]["s3_bucket"]

            s3_key = (
                f"{self.config['aws']['s3_model_prefix']}{symbol}/model_{run_id}.txt"
            )
            model_path = f"/tmp/model_{symbol}_{run_id}.txt"

            s3_client.download_file(bucket, s3_key, model_path)
            self.model = lgb.Booster(model_file=model_path)

            logger.info(f"Loaded model from S3: s3://{bucket}/{s3_key}")
            os.remove(model_path)

        except Exception as e:
            logger.error(f"Error loading model from S3: {str(e)}")


# Usage example
if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from data_processing import CryptoDataProcessor

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Prepare data
    processor = CryptoDataProcessor(config)
    symbol = "BTC-USD"
    data = processor.prepare_training_data(symbol)

    # Train model
    model = CryptoLightGBMModel(config)
    results = model.train_model(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        symbol,
        optimize_params=True,
    )

    print(f"Training completed with validation R²: {results['metrics']['val_r2']:.4f}")
    print(f"Top 10 important features:")
    print(results["feature_importance"].head(10))