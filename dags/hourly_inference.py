import io
import logging
from datetime import datetime, timedelta

import boto3
import mlflow
import mlflow.lightgbm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator

# === Tes modules ===
from src.data_ingestion import CoinbaseDataClient
from src.data_processing import CryptoDataProcessor

# === Config Airflow ===
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with open("/opt/airflow/src/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S3_ENDPOINT = "http://localstack:4566"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
BUCKET_NAME = CONFIG["aws"]["s3_bucket"]
SYMBOL = CONFIG["data"]["symbols"][0]

dag = DAG(
    "preprocessing_hourly",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
)

# === S3 Client ===
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)


def create_bucket_if_needed():
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        logging.info(f"Bucket {BUCKET_NAME} exists")
    except s3_client.exceptions.ClientError:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        logging.info(f"Bucket {BUCKET_NAME} created")


def preprocess_and_predict(**context):
    execution_date = context["execution_date"]

    if isinstance(execution_date, str):
        execution_date = datetime.fromisoformat(execution_date)

    logging.info(
        f"[DEBUG] Execution date: {execution_date} (type={type(execution_date)})"
    )

    date_str = execution_date.strftime("%Y-%m-%d")
    hour_str = execution_date.strftime("%H")

    create_bucket_if_needed()

    # --- Data ingestion ---
    client = CoinbaseDataClient(config=CONFIG)
    df_raw = client.fetch_and_save_historical(
        execution_date, days_back=CONFIG["data"]["days_back"]
    )
    logging.info(f"Données brutes récupérées : shape={df_raw.shape}")

    processor = CryptoDataProcessor(config=CONFIG)
    df_raw_combined = processor.combine_datasets(SYMBOL)
    logging.info(f"Données combinées : shape={df_raw_combined.shape}")

    df_features = processor.engineer_features(df_raw_combined)
    logging.info(f"Features : shape={df_features.shape}")

    scaler_filename = f"scaler_{SYMBOL.replace('-', '_')}.pickle"
    processor.load_preprocessor(scaler_filename)

    X, y = processor.prepare_features_target(df_features)
    X_scaled = processor.scaler.transform(X)
    df_features_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    last_ts = df_features_scaled.index.max()
    feature_index = last_ts
    df_features = df_features_scaled.loc[[feature_index]]
    df_last = df_features_scaled.loc[[last_ts]]
    df_last["target"] = y.loc[last_ts]

    logging.info(f"Last row : shape={df_features.shape}")

    # --- Save features ---
    table = pa.Table.from_pandas(df_features.reset_index())
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)

    key_features = f"features/date={date_str}/hour={hour_str}/features.parquet"
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key_features, Body=buffer.getvalue())
    logging.info(f"✅ Features uploaded → s3://{BUCKET_NAME}/{key_features}")

    # --- Save target ---
    df_target = pd.DataFrame({"target": [df_last["target"].iloc[0]]})
    table_target = pa.Table.from_pandas(df_target.reset_index(drop=True))
    buffer_target = io.BytesIO()
    pq.write_table(table_target, buffer_target)
    buffer_target.seek(0)

    key_target = f"targets/date={date_str}/hour={hour_str}/target.parquet"
    s3_client.put_object(
        Bucket=BUCKET_NAME, Key=key_target, Body=buffer_target.getvalue()
    )
    logging.info(f"✅ Target uploaded → s3://{BUCKET_NAME}/{key_target}")

    # === Load model like API ===
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    model_name = f"{CONFIG['mlflow']['model_name']}_{SYMBOL.replace('-', '_')}"
    rm = client.search_registered_models(f"name='{model_name}'")[0]
    prod_version_number = rm.aliases.get("production")
    if not prod_version_number:
        raise Exception(f"No 'production' alias for {model_name}")

    model = mlflow.lightgbm.load_model(f"models:/{model_name}/{prod_version_number}")
    logging.info(f"✅ Model loaded: {model_name}, version {prod_version_number}")

    # === Predict ===
    df_last_for_pred = df_last.drop(columns=["target"], errors="ignore")
    cols_to_drop = ["timestamp", "symbol", "index"]
    df_last_for_pred = df_last_for_pred.drop(
        columns=[col for col in cols_to_drop if col in df_last_for_pred.columns],
        errors="ignore",
    )

    y_pred = model.predict(df_last_for_pred)

    df_prediction = pd.DataFrame({"prediction": [y_pred[0]]})
    table_pred = pa.Table.from_pandas(df_prediction.reset_index(drop=True))
    buffer_pred = io.BytesIO()
    pq.write_table(table_pred, buffer_pred)
    buffer_pred.seek(0)

    key_pred = f"predictions/date={date_str}/hour={hour_str}/prediction.parquet"
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key_pred, Body=buffer_pred.getvalue())
    logging.info(f"✅ Prediction uploaded → s3://{BUCKET_NAME}/{key_pred}")




with dag:
    task = PythonOperator(
        task_id="preprocess_and_predict",
        python_callable=preprocess_and_predict,
        provide_context=True,
    )
