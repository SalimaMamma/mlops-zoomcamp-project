from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.model import CryptoLightGBMModel


class TestCryptoLightGBMModel:

    def test_init(self, config):
        """Test model initialization"""
        model = CryptoLightGBMModel(config)
        assert model.config == config
        assert model.model is None
        assert model.model_config == config["model"]

    def test_create_lgb_datasets(self, config):
        """Test LightGBM dataset creation"""
        model = CryptoLightGBMModel(config)

        # Create sample data
        X_train = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(
            np.random.randn(20, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y_val = pd.Series(np.random.randn(20))

        train_data, val_data = model.create_lgb_datasets(X_train, y_train, X_val, y_val)

        assert isinstance(train_data, lgb.Dataset)
        assert isinstance(val_data, lgb.Dataset)

    @patch("mlflow.start_run")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.lightgbm.log_model")
    @patch("mlflow.active_run")
    def test_train_model(
        self,
        mock_active_run,
        mock_log_model,
        mock_log_metrics,
        mock_log_params,
        mock_start_run,
        config,
    ):
        """Test model training"""
        model = CryptoLightGBMModel(config)

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_active_run.return_value = mock_run
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Create sample data
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(
            np.random.randn(20, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        y_val = pd.Series(np.random.randn(20))

        symbol = "BTC-USD"
        results = model.train_model(
            X_train, y_train, X_val, y_val, symbol, optimize_params=False
        )

        # Check that model was trained
        assert model.model is not None

        # Check results structure
        expected_keys = ["model", "metrics", "feature_importance", "run_id"]
        for key in expected_keys:
            assert key in results

        # Check metrics
        metrics = results["metrics"]
        assert "train_rmse" in metrics
        assert "val_rmse" in metrics
        assert "train_r2" in metrics
        assert "val_r2" in metrics

        # Check feature importance
        assert isinstance(results["feature_importance"], pd.DataFrame)
        assert "feature" in results["feature_importance"].columns
        assert "importance" in results["feature_importance"].columns

    def test_predict(self, config):
        """Test model prediction"""
        model = CryptoLightGBMModel(config)

        # Test prediction without trained model
        X_test = pd.DataFrame(np.random.randn(10, 5))
        with pytest.raises(ValueError):
            model.predict(X_test)

        # Mock a trained model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randn(10)
        model.model = mock_model

        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)

    @patch("boto3.client")
    @patch("os.remove")
    def test_save_model_to_s3(self, mock_remove, mock_boto_client, config):
        """Test saving model to S3"""
        model = CryptoLightGBMModel(config)

        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock trained model
        mock_lgb_model = MagicMock()
        model.model = mock_lgb_model

        # Mock feature importance
        model.feature_importance = pd.DataFrame(
            {"feature": ["feat1", "feat2"], "importance": [0.6, 0.4]}
        )

        symbol = "BTC-USD"
        run_id = "test_run"

        # This should not raise an exception
        model.save_model_to_s3(symbol, run_id)

        # Verify S3 put_object was called
        assert mock_s3.put_object.call_count >= 1
