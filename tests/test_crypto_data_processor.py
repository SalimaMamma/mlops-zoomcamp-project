import io
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_processing import CryptoDataProcessor


class TestCryptoDataProcessor:

    def test_init(self, config):
        """Test processor initialization"""
        processor = CryptoDataProcessor(config)
        assert processor.config == config
        assert processor.bucket == config["aws"]["s3_bucket"]
        assert processor.lookback_periods == config["data"]["lookback_periods"]

    def test_load_data_from_s3(self, config, mock_s3, sample_ohlcv_data):
        """Test loading data from S3"""
        processor = CryptoDataProcessor(config)
        processor.s3_client = mock_s3

        # Save test data to S3
        csv_content = sample_ohlcv_data.to_csv(index=False)
        s3_key = "test_data.csv"
        mock_s3.put_object(
            Bucket=config["aws"]["s3_bucket"], Key=s3_key, Body=csv_content
        )

        # Load data
        df = processor.load_data_from_s3(s3_key)

        assert not df.empty
        assert len(df) == len(sample_ohlcv_data)
        assert "timestamp" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_data_from_s3_missing_file(self, config, mock_s3):
        """Test loading non-existent file from S3"""
        processor = CryptoDataProcessor(config)
        processor.s3_client = mock_s3

        df = processor.load_data_from_s3("non_existent.csv")

        assert df.empty

    def test_list_data_files(self, config, mock_s3):
        """Test listing data files in S3"""
        processor = CryptoDataProcessor(config)
        processor.s3_client = mock_s3

        # Create test files
        test_files = [
            "data/btc_usd_20230101.csv",
            "data/eth_usd_20230101.csv",
            "data/other_file.txt",
        ]

        for file in test_files:
            mock_s3.put_object(
                Bucket=config["aws"]["s3_bucket"], Key=file, Body="test content"
            )

        # Test listing all files
        files = processor.list_data_files()
        assert len(files) == 3

        # Test filtering by symbol
        btc_files = processor.list_data_files("BTC-USD")
        assert len(btc_files) == 1
        assert "btc_usd" in btc_files[0].lower()

    def test_engineer_features(self, config, sample_ohlcv_data):
        """Test feature engineering"""
        processor = CryptoDataProcessor(config)

        df_features = processor.engineer_features(sample_ohlcv_data)

        # Check basic features
        assert "price_change" in df_features.columns
        assert "volume_change" in df_features.columns
        assert "high_low_ratio" in df_features.columns

        # Check moving averages
        for period in config["data"]["lookback_periods"]:
            assert f"sma_{period}" in df_features.columns
            assert f"ema_{period}" in df_features.columns

        # Check technical indicators
        assert "rsi" in df_features.columns
        assert "macd" in df_features.columns
        assert "atr" in df_features.columns

        # Check lag features
        assert "close_lag_1" in df_features.columns
        assert "volume_lag_5" in df_features.columns

        # Check time features
        assert "hour" in df_features.columns
        assert "day_of_week" in df_features.columns

        # Check target variable
        assert "target" in df_features.columns
        assert "target_change" in df_features.columns

        # No NaN values should remain
        assert not df_features.isnull().any().any()

    def test_prepare_features_target(self, config, sample_ohlcv_data):
        """Test feature and target preparation"""
        processor = CryptoDataProcessor(config)
        df_features = processor.engineer_features(sample_ohlcv_data)

        X, y = processor.prepare_features_target(df_features)

        # Check shapes
        assert len(X) == len(y)
        assert len(X) > 0

        # Check that excluded columns are not in features
        excluded = ["timestamp", "symbol", "target", "target_change"]
        for col in excluded:
            assert col not in X.columns

        # Check no infinite values
        assert not np.isinf(X.values).any()
        assert not np.isinf(y.values).any()

    def test_save_load_preprocessor(self, config, mock_s3, sample_ohlcv_data):
        """Test saving and loading preprocessor"""
        processor = CryptoDataProcessor(config)
        processor.s3_client = mock_s3

        # Prepare some data to fit scaler
        df_features = processor.engineer_features(sample_ohlcv_data)
        X, y = processor.prepare_features_target(df_features)
        processor.scaler.fit(X)

        # Save preprocessor
        filename = "test_scaler.pickle"
        processor.save_preprocessor(filename)

        # Create new processor and load
        new_processor = CryptoDataProcessor(config)
        new_processor.s3_client = mock_s3
        new_processor.load_preprocessor(filename)

        # Test that loaded scaler works
        X_scaled = new_processor.scaler.transform(X)
        assert X_scaled.shape == X.shape

    def test_clean_data(self, config, sample_ohlcv_data):
        """Test data cleaning"""
        processor = CryptoDataProcessor(config)

        # Add some NaN values
        dirty_data = sample_ohlcv_data.copy()
        dirty_data.loc[0:5, "close"] = np.nan
        dirty_data.loc[10:15, "volume"] = np.nan

        clean_data = processor._clean_data(dirty_data)

        # Should have fewer rows
        assert len(clean_data) <= len(dirty_data)

        # Should have no NaN values
        assert not clean_data.isnull().any().any()

    @patch("src.data_processing.CryptoDataProcessor.combine_datasets")
    def test_prepare_training_data(self, mock_combine, config, sample_ohlcv_data):
        """Test complete training data preparation"""
        processor = CryptoDataProcessor(config)
        processor.s3_client = MagicMock()

        # Mock the combine_datasets method
        mock_combine.return_value = sample_ohlcv_data

        symbol = "BTC-USD"
        data = processor.prepare_training_data(symbol, test_size=0.2)

        # Check returned structure
        expected_keys = [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "feature_names",
            "preprocessor_filename",
            "train_dates",
            "test_dates",
        ]
        for key in expected_keys:
            assert key in data

        # Check split proportions
        total_samples = len(data["X_train"]) + len(data["X_test"])
        test_ratio = len(data["X_test"]) / total_samples
        assert 0.15 <= test_ratio <= 0.25  # Allow some tolerance

        # Check that features are scaled
        assert data["X_train"].mean().abs().max() < 5
