import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_data_ingestion_to_processing_pipeline(self, config, mock_s3):
        """Test pipeline from data ingestion to feature engineering"""
        from src.data_ingestion import CoinbaseDataClient
        from src.data_processing import CryptoDataProcessor

        # Réduire les périodes de lookback pour éviter l'erreur d'index
        config["data"]["lookback_periods"] = [5, 10]

        # Mock API response avec BEAUCOUP plus de données (100 points)
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()

            # Générer 100 points de données au lieu de 3
            mock_data = []
            base_timestamp = 1640995200  # 1er janvier 2022
            for i in range(100):
                timestamp = base_timestamp + (i * 300)  # 5 minutes d'intervalle
                open_price = 46000 + (i * 10)
                high_price = open_price + 200
                low_price = open_price - 200
                close_price = open_price + 50
                volume = 1000 + (i * 10)
                mock_data.append(
                    [timestamp, low_price, high_price, open_price, close_price, volume]
                )

            mock_response.json.return_value = mock_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Ingest data
            client = CoinbaseDataClient(config)
            client.s3_client = mock_s3
            df = client.fetch_and_save_historical(days_back=1)

            # Process data
            processor = CryptoDataProcessor(config)
            processor.s3_client = mock_s3

            # Mock list_data_files to return our saved file
            with patch.object(processor, "list_data_files") as mock_list:
                mock_list.return_value = ["data/historical_BTC_USD_test.csv"]

                # Mock load_data_from_s3 to return our ingested data
                with patch.object(processor, "load_data_from_s3", return_value=df):
                    combined_data = processor.combine_datasets("BTC-USD")

                    assert not combined_data.empty
                    assert "symbol" in combined_data.columns
                    assert len(combined_data) == 100  # Vérifier qu'on a bien 100 points

                    # Engineer features
                    features_df = processor.engineer_features(combined_data)

                    # Should have many more columns after feature engineering
                    assert features_df.shape[1] > combined_data.shape[1]
                    assert "target" in features_df.columns

                    assert len(features_df) > 20
