from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_ingestion import CoinbaseDataClient


@pytest.fixture
def config():
    return {
        "data": {
            "coinbase_api_url": "https://api.pro.coinbase.com",
            "websocket_url": "",
            "symbols": ["BTC-USD"],
        },
        "aws": {
            "region": "us-east-1",
            "access_key_id": "fake",
            "secret_access_key": "fake",
            "endpoint_url": "https://s3.fake",
            "s3_bucket": "fake-bucket",
            "s3_data_prefix": "test/",
        },
    }


@pytest.fixture
def client(config):
    return CoinbaseDataClient(config)


@patch("requests.get")
def test_get_historical_data_range_success(mock_get, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        [1625097600, 34000, 35000, 34500, 34800, 12],
        [1625101200, 34800, 35500, 34850, 35000, 10],
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 1, 1)

    df = client.get_historical_data_range("BTC-USD", start, end)

    assert not df.empty
    assert "timestamp" in df.columns
    assert all(df["volume"] > 0)
    assert (df["symbol"] == "BTC-USD").all()


@patch("requests.get")
def test_get_historical_data_range_empty(mock_get, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 1, 1)

    df = client.get_historical_data_range("BTC-USD", start, end)

    assert df.empty


@patch("requests.get")
def test_get_historical_data_range_exception(mock_get, client):
    mock_get.side_effect = Exception("API Down")

    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 1, 1)

    df = client.get_historical_data_range("BTC-USD", start, end)

    assert df.empty


def test_save_to_s3_success(client):
    with patch.object(client, "s3_client") as mock_s3:
        df = pd.DataFrame({"a": [1], "b": [2]})

        client.save_to_s3(df, "test.csv")

        mock_s3.put_object.assert_called_once()


def test_save_to_s3_empty(client):
    with patch.object(client, "s3_client") as mock_s3:
        df = pd.DataFrame()

        client.save_to_s3(df, "test.csv")

        mock_s3.put_object.assert_not_called()


def test_fetch_and_save_historical(client):
    with patch.object(client, "get_historical_data_range") as mock_get, patch.object(
        client, "save_to_s3"
    ) as mock_save:

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "low": [1],
                "high": [2],
                "open": [1.5],
                "close": [1.8],
                "volume": [10],
            }
        )
        mock_get.return_value = df

        result_df = client.fetch_and_save_historical(days_back=1)

        mock_get.assert_called()
        mock_save.assert_called()
        assert not result_df.empty
