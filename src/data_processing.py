import io
import logging
import pickle
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import ta
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class CryptoDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = boto3.client(
            "s3",
            region_name=config["aws"]["region"],
            aws_access_key_id=config["aws"]["access_key_id"],
            aws_secret_access_key=config["aws"]["secret_access_key"],
            endpoint_url=config["aws"]["endpoint_url"],
        )
        self.bucket = config["aws"]["s3_bucket"]
        self.scaler = StandardScaler()
        self.lookback_periods = config["data"]["lookback_periods"]
        self.features = config["data"]["features"]

    def load_data_from_s3(self, s3_key: str) -> pd.DataFrame:
        """Load data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            csv_content = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_content))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(f"Loaded {len(df)} records from S3: {s3_key}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from S3: {str(e)}")
            return pd.DataFrame()

    def list_data_files(self, symbol: Optional[str] = None) -> List[str]:
        """List all data files in S3"""
        prefix = self.config["aws"]["s3_data_prefix"]

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", [])]

            if symbol:
                symbol_clean = symbol.replace("-", "_").lower()
                files = [f for f in files if symbol_clean in f.lower()]

            return files
        except Exception as e:
            logger.error(f"Error listing S3 files: {str(e)}")
            return []

    def combine_datasets(self, symbol: str) -> pd.DataFrame:
        """Combine all datasets for a symbol"""
        files = self.list_data_files(symbol)

        if not files:
            logger.warning(f"No files found for symbol: {symbol}")
            return pd.DataFrame()

        dfs = []
        for file in files:
            df = self.load_data_from_s3(file)
            if not df.empty:
                dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=["timestamp", "symbol"]
            ).sort_values("timestamp")
            combined_df = combined_df.reset_index(drop=True)
            logger.info(f"Combined {len(combined_df)} records for {symbol}")
            return combined_df

        return pd.DataFrame()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Basic price features
        df["price_change"] = df["close"].pct_change()
        df["volume_change"] = df["volume"].pct_change()
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]
        df["price_range"] = (df["high"] - df["low"]) / df["open"]

        # Moving averages
        for period in self.lookback_periods:
            df[f"sma_{period}"] = ta.trend.sma_indicator(df["close"], window=period)
            df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], window=period)
            df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()

            # Price relative to moving averages
            df[f"price_vs_sma_{period}"] = df["close"] / df[f"sma_{period}"] - 1
            df[f"price_vs_ema_{period}"] = df["close"] / df[f"ema_{period}"] - 1

        # Technical indicators using ta library
        # Trend indicators
        df["macd"] = ta.trend.macd_diff(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])

        # Momentum indicators
        df["rsi"] = ta.momentum.rsi(df["close"])

        # Volatility indicators
        df["bbands_upper"] = ta.volatility.bollinger_hband(df["close"])
        df["bbands_lower"] = ta.volatility.bollinger_lband(df["close"])
        df["bbands_position"] = (df["close"] - df["bbands_lower"]) / (
            df["bbands_upper"] - df["bbands_lower"]
        )

        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

        # Volume indicators
        df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
            df[f"price_change_lag_{lag}"] = df["price_change"].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"close_std_{window}"] = df["close"].rolling(window=window).std()
            df[f"volume_std_{window}"] = df["volume"].rolling(window=window).std()
            df[f"price_change_std_{window}"] = (
                df["price_change"].rolling(window=window).std()
            )

            # Rolling min/max
            df[f"close_min_{window}"] = df["close"].rolling(window=window).min()
            df[f"close_max_{window}"] = df["close"].rolling(window=window).max()
            df[f"close_position_{window}"] = (
                df["close"] - df[f"close_min_{window}"]
            ) / (df[f"close_max_{window}"] - df[f"close_min_{window}"])

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month

        # Target variable (next period close price)
        df["target"] = df["close"].shift(-self.config["data"]["prediction_horizon"])
        df["target_change"] = (
            df["target"] / df["close"] - 1
        ) * 100  # Percentage change

        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(
            f"Removed {initial_rows - len(df)} rows with NaN values. Final shape: {df.shape}"
        )

        return df

    def prepare_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        # Exclude non-feature columns
        exclude_cols = ["timestamp", "symbol", "target", "target_change"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df["target_change"].copy()  # Predict percentage change

        # Handle any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method="ffill").fillna(method="bfill")

        logger.info(f"Prepared features: {X.shape}, Target: {y.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")

        return X, y

    import io

    def save_preprocessor(self, filename: str):
        """Save scaler to S3 using joblib"""
        try:
            buffer = io.BytesIO()
            joblib.dump(self.scaler, buffer)
            buffer.seek(0)

            s3_key = f"{self.config['aws']['s3_model_prefix']}preprocessors/{filename}"

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
            )
            logger.info(f"Saved preprocessor to S3: s3://{self.bucket}/{s3_key}")

        except Exception as e:
            logger.error(f"Error saving preprocessor to S3: {str(e)}")

    def load_preprocessor(self, filename: str):
        """Load scaler from S3"""
        try:

            s3_key = f"{self.config['aws']['s3_model_prefix']}preprocessors/{filename}"
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            scaler_data = response["Body"].read()
            buffer = BytesIO(scaler_data)
            buffer.seek(0)
            self.scaler = joblib.load(buffer)
            self.features = list(self.scaler.feature_names_in_)

        except Exception as e:
            logger.error(f"Error loading preprocessor from S3: {str(e)}")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        try:
            # Remove rows with too many NaN values
            data = data.dropna(thresh=len(data.columns) * 0.7)

            # Forward fill remaining NaN values
            data = data.fillna(method="ffill")

            # Remove any remaining NaN values
            data = data.dropna()

            """
            # Remove outliers (optional - using IQR method)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['volume']:  # Skip volume outlier removal
                    continue
                    
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = np.clip(data[col], lower, upper)
            """

            return data

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return data

    def prepare_training_data(self, symbol: str, test_size: float = 0.2) -> Dict:
        """Complete data preparation pipeline"""
        # Load and combine data
        df = self.combine_datasets(symbol)
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Engineer features
        df_processed = self.engineer_features(df)
        df_cleaned = self._clean_data(df_processed)

        if len(df_processed) < 100:
            raise ValueError(
                f"Insufficient data after processing: {len(df_processed)} rows"
            )

        # Prepare features and target
        X, y = self.prepare_features_target(df_cleaned)

        # Split data (time series split - no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Save preprocessor
        preprocessor_filename = f"scaler_{symbol.replace('-', '_')}.pickle"
        self.save_preprocessor(preprocessor_filename)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(X.columns),
            "preprocessor_filename": preprocessor_filename,
            "train_dates": df_processed.iloc[:split_idx]["timestamp"],
            "test_dates": df_processed.iloc[split_idx:]["timestamp"],
        }


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    processor = CryptoDataProcessor(config)

    for symbol in config["data"]["symbols"]:
        try:
            data = processor.prepare_training_data(symbol)
            print(f"Prepared training data for {symbol}")
            print(f"Training set shape: {data['X_train'].shape}")
            print(f"Test set shape: {data['X_test'].shape}")
            print(f"Features: {len(data['feature_names'])}")
        except Exception as e:
            print(f"Error preparing data for {symbol}: {str(e)}")
