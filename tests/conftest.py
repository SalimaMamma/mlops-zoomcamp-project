import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import boto3
import numpy as np
import pandas as pd
import pytest
import yaml
from moto import mock_aws


@pytest.fixture
def config():
    """Configuration fixture for tests"""
    return {
        "aws": {
            "region": "us-east-1",
            "access_key_id": "test_key",
            "secret_access_key": "test_secret",
            "endpoint_url": "http://localhost:9000",
            "s3_bucket": "test-bucket",
            "s3_data_prefix": "data/",
            "s3_model_prefix": "models/",
        },
        "data": {
            "coinbase_api_url": "https://api.exchange.coinbase.com",
            "websocket_url": "wss://ws-feed.exchange.coinbase.com",
            "symbols": ["BTC-USD", "ETH-USD"],
            "lookback_periods": [5, 10, 20],
            "features": ["close", "volume", "high", "low"],
            "prediction_horizon": 1,
        },
        "model": {
            "objective": "regression",
            "metric": "rmse",
            "num_boost_round": 100,
            "early_stopping_rounds": 10,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "model_name": "crypto_predictor",
        },
    }


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=8640, freq="5T")
    n = len(dates)

    np.random.seed(42)
    base_price = 45000

    data = {
        "timestamp": dates,
        "open": base_price + np.random.normal(0, 100, n),
        "high": base_price + np.random.normal(200, 100, n),
        "low": base_price + np.random.normal(-200, 100, n),
        "close": base_price + np.random.normal(0, 100, n),
        "volume": np.random.lognormal(10, 1, n),
        "symbol": ["BTC-USD"] * n,
    }

    df = pd.DataFrame(data)
    # Ensure high >= low
    df["high"] = np.maximum(df[["open", "close", "high"]].max(axis=1), df["high"])
    df["low"] = np.minimum(df[["open", "close", "low"]].min(axis=1), df["low"])

    return df


@pytest.fixture
def mock_s3():
    """Mock S3 service"""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield s3
