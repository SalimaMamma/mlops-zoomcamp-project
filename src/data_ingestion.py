import logging
import queue
from datetime import datetime, timedelta
from typing import Dict

import boto3
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class CoinbaseDataClient:
    def __init__(self, config: Dict):
        self.config = config
        self.api_url = config["data"]["coinbase_api_url"]
        self.websocket_url = config["data"]["websocket_url"]
        self.symbols = config["data"]["symbols"]
        self.s3_client = boto3.client(
            "s3",
            region_name=config["aws"]["region"],
            aws_access_key_id=config["aws"]["access_key_id"],
            aws_secret_access_key=config["aws"]["secret_access_key"],
            endpoint_url=config["aws"]["endpoint_url"],
        )
        self.bucket = config["aws"]["s3_bucket"]
        self.data_queue = queue.Queue()

    def get_historical_data_range(
        self, symbol: str, start: datetime, end: datetime, granularity: int = 300
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data in chunks to stay within Coinbase API candle limits.
        """
        all_data = []
        max_candles = 300
        delta = timedelta(seconds=granularity * max_candles)
        current_start = start

        while current_start < end:
            current_end = min(current_start + delta, end)
            logger.info(f"current_start: {current_start}, current_end: {current_end}")

            url = f"{self.api_url}/products/{symbol}/candles"
            params = {
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
                "granularity": granularity,
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data:
                    df = pd.DataFrame(
                        data,
                        columns=["timestamp", "low", "high", "open", "close", "volume"],
                    )
                    all_data.append(df)
                    logger.info(
                        f"Fetched {len(df)} candles for {symbol} from {current_start} to {current_end}"
                    )
                else:
                    logger.warning(
                        f"No data for {symbol} from {current_start} to {current_end}"
                    )

            except Exception as e:
                logger.error(
                    f"Error fetching {symbol} from {current_start} to {current_end}: {e}"
                )

            current_start = current_end

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], unit="s")
            df_all = df_all.sort_values("timestamp").reset_index(drop=True)
            df_all["symbol"] = symbol
            df_all = df_all.dropna()
            df_all = df_all[df_all["volume"] > 0]  # drop rows with zero volume
            return df_all
        else:
            logger.warning(f"No data collected for {symbol} in given range.")
            return pd.DataFrame()

    def save_to_s3(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to S3.
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame, skipping save for {filename}")
                return

            csv_buffer = df.to_csv(index=False)
            s3_key = f"{self.config['aws']['s3_data_prefix']}{filename}"

            self.s3_client.put_object(
                Bucket=self.bucket, Key=s3_key, Body=csv_buffer, ContentType="text/csv"
            )
            logger.info(f"Saved data to S3: s3://{self.bucket}/{s3_key}")

        except Exception as e:
            logger.error(f"Error saving to S3: {e}")

    def fetch_and_save_historical(self, end_date=None, days_back: int = 30):
        """
        Fetch and save historical data for all symbols.
        """
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for symbol in self.symbols:
            try:
                df = self.get_historical_data_range(
                    symbol,
                    start=start_date,
                    end=end_date,
                    granularity=300,  # 5-minute candles
                )

                if not df.empty:
                    filename = f"historical_{symbol.replace('-', '_')}_{end_date.strftime('%Y%m%d%H%M%S')}.csv"
                    self.save_to_s3(df, filename)
                else:
                    logger.warning(f"No data to save for {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        return df


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    client = CoinbaseDataClient(config)
    client.fetch_and_save_historical(days_back=60)
