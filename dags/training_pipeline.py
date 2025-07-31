import logging
import math
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import boto3
import mlflow
import numpy as np
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your actual modules
from src.data_ingestion import CoinbaseDataClient
from src.data_processing import CryptoDataProcessor
from src.model import CryptoLightGBMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("/opt/airflow/src/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

s3_client = boto3.client(
    "s3",
    endpoint_url=CONFIG["aws"]["endpoint_url"],
    aws_access_key_id=CONFIG["aws"]["access_key_id"],
    aws_secret_access_key=CONFIG["aws"]["secret_access_key"],
    region_name=CONFIG["aws"]["region"],
)

BUCKET_NAME = CONFIG["aws"]["s3_bucket"]


# 🗂️ Fonction pour créer le bucket si absent
def create_bucket_if_not_exists():
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        logger.info(f"Bucket {BUCKET_NAME} déjà existant ✅")
    except s3_client.exceptions.ClientError:
        logger.info(f"Bucket {BUCKET_NAME} inexistant, création...")
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        logger.info(f"Bucket {BUCKET_NAME} créé ✅")


SYMBOL = CONFIG["data"]["symbols"][0]
DRIFT_THRESHOLD = CONFIG["model"]["mae_drift_threshold"]

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email": ["mlops_team@example.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
}


def validate_and_clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Valide et nettoie les métriques pour éviter les erreurs de sérialisation MLflow
    """
    cleaned_metrics = {}

    for key, value in metrics.items():
        try:
           
            float_value = float(value)

            
            if math.isnan(float_value) or math.isinf(float_value):
                logger.warning(
                    f"Métrique {key} invalide ({float_value}), remplacée par 0.0"
                )
                cleaned_metrics[key] = 0.0
            else:
                
                cleaned_metrics[key] = round(float_value, 6)

        except (ValueError, TypeError) as e:
            logger.warning(f"Impossible de convertir la métrique {key}={value}: {e}")
            cleaned_metrics[key] = 0.0

    return cleaned_metrics


def safe_get_metric(run_data, metric_name: str, default: float = float("inf")) -> float:
    """
    Récupère une métrique de manière sécurisée
    """
    try:
        value = run_data.metrics.get(metric_name, default)
        if math.isnan(value) or math.isinf(value):
            logger.warning(
                f"Métrique {metric_name} invalide, utilisation de la valeur par défaut"
            )
            return default
        return value
    except Exception as e:
        logger.error(
            f"Erreur lors de la récupération de la métrique {metric_name}: {e}"
        )
        return default


# Task 0: Create bucket
def create_bucket_task():
    create_bucket_if_not_exists()


# Task 1: Data Ingestion
def ingest():
    client = CoinbaseDataClient(CONFIG)
    client.fetch_and_save_historical(days_back=1)


# Task 2: Data Processing & Feature Engineering
def process():
    processor = CryptoDataProcessor(CONFIG)
    data = processor.prepare_training_data(SYMBOL)


# Task 3: Model Training + Tracking + Save to S3
def train():
    processor = CryptoDataProcessor(CONFIG)
    data = processor.prepare_training_data(SYMBOL)

    model = CryptoLightGBMModel(CONFIG)
    result = model.train_model(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"], SYMBOL
    )

    # Validation et nettoyage des métriques avant sauvegarde
    if "metrics" in result:
        cleaned_metrics = validate_and_clean_metrics(result["metrics"])
        logger.info(f"Métriques nettoyées: {cleaned_metrics}")

    model.save_model_to_s3(SYMBOL, result["run_id"])


# Task 4: Promote model to Production with robust error handling
def promote_model():
    import os
    import time

    max_connection_retries = 5
    for attempt in range(max_connection_retries):
        try:
            mlflow_urls = [
                "http://mlflow:5000",
                "http://localhost:5000",
                os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            ]
            client = None
            for url in mlflow_urls:
                try:
                    mlflow.set_tracking_uri(url)
                    client = mlflow.tracking.MlflowClient()
                    client.search_experiments(max_results=1)
                    logger.info(f"✅ Connexion MLflow OK: {url}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Échec connexion {url}: {e}")
                    continue

            if not client:
                raise Exception("Aucune URL MLflow accessible")
            break

        except Exception as e:
            logger.error(f"Tentative {attempt+1}/{max_connection_retries} : {e}")
            if attempt < max_connection_retries - 1:
                time.sleep(30)
            else:
                logger.error("❌ Impossible de se connecter à MLflow")
                return

    model_name = f"{CONFIG['mlflow']['model_name']}_{SYMBOL.replace('-', '_')}"
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        logger.warning("⚠️ Aucun modèle trouvé")
        return

    all_versions.sort(key=lambda x: int(x.version), reverse=True)
    latest_version = all_versions[0]

    logger.info(f"👉 Promotion version {latest_version.version}")

    # ✅ Méthode RECOMMANDÉE : Alias
    client.set_registered_model_alias(
        name=model_name, alias="production", version=latest_version.version
    )
    logger.info(
        f"✅ Modèle promu : alias 'production' pointé vers version {latest_version.version}"
    )


# Define DAG
with DAG(
    dag_id="training_pipeline",
    description="Complete pipeline: Ingest → Process → Train → Promote → Drift Monitoring",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    task_create_bucket = PythonOperator(
        task_id="create_bucket", python_callable=create_bucket_task
    )

    task_ingest = PythonOperator(task_id="ingest_data", python_callable=ingest)

    task_process = PythonOperator(task_id="process_data", python_callable=process)

    task_train = PythonOperator(task_id="train_model", python_callable=train)

    task_promote = PythonOperator(
        task_id="promote_model", python_callable=promote_model
    )

    # Define task dependencies
    task_create_bucket >> task_ingest >> task_process >> task_train >> task_promote
