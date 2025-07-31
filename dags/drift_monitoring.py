import os
import io
import logging
from datetime import datetime, timedelta, timezone
import json
import boto3
import pandas as pd
import pyarrow.parquet as pq
import yaml

from airflow import DAG
from airflow.api.common.experimental.trigger_dag import trigger_dag
from airflow.operators.python import PythonOperator
from botocore.exceptions import ClientError
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import RegressionPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    logger.warning("Evidently package not found, skipping report generation")
    EVIDENTLY_AVAILABLE = False

# === Config ===
with open("/opt/airflow/src/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S3_ENDPOINT = CONFIG["aws"]["endpoint_url"]
AWS_ACCESS_KEY_ID = CONFIG["aws"]["access_key_id"]
AWS_SECRET_ACCESS_KEY = CONFIG["aws"]["secret_access_key"]
BUCKET_NAME = CONFIG["aws"]["s3_bucket"]
SYMBOL = CONFIG["data"]["symbols"][0]

REPORT_DIR = "/opt/airflow/dags/reports"
DRIFT_THRESHOLD = CONFIG["monitoring"]["drift_threshold"]

os.makedirs(REPORT_DIR, exist_ok=True)

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)

def load_parquet(bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
        return df
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"File not found: s3://{bucket}/{key}")
            return None
        else:
            logger.error(f"Error loading file s3://{bucket}/{key}: {e}")
            raise

def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame | None:
    """Charge un fichier CSV depuis S3"""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"CSV file not found: s3://{bucket}/{key}")
            return None
        else:
            logger.error(f"Error loading CSV file s3://{bucket}/{key}: {e}")
            raise

def get_latest_run_id(bucket: str, symbol: str) -> str | None:
    """Récupère le dernier run_id en listant les fichiers de référence"""
    try:
        prefix = f"{CONFIG['aws']['s3_model_prefix']}/reference/"
        logger.info(f"Searching for reference files with prefix: {prefix}")
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No files found with prefix: {prefix}")
            # Essaie de lister le répertoire parent pour voir ce qui existe
            try:
                parent_prefix = f"{CONFIG['aws']['s3_model_prefix']}/"
                logger.info(f"Checking parent directory: {parent_prefix}")
                parent_response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=parent_prefix
                )
                if 'Contents' in parent_response:
                    logger.info("Files in parent directory:")
                    for obj in parent_response['Contents'][:10]:  # Limite à 10 pour éviter trop de logs
                        logger.info(f"  - {obj['Key']}")
            except Exception as e:
                logger.error(f"Error listing parent directory: {e}")
            return None
        
        logger.info(f"Found {len(response['Contents'])} files in reference directory")
        
        # Cherche les fichiers train_*.csv et extrait les run_ids
        run_ids = []
        for obj in response['Contents']:
            key = obj['Key']
            logger.debug(f"Examining file: {key}")
            if key.endswith('.csv') and 'train_' in key:
                # Extrait le run_id du nom de fichier
                filename = key.split('/')[-1]  # train_run_id.csv
                if filename.startswith('train_'):
                    run_id = filename.replace('train_', '').replace('.csv', '')
                    run_ids.append((run_id, obj['LastModified']))
                    logger.info(f"Found run_id: {run_id} (modified: {obj['LastModified']})")
        
        if not run_ids:
            logger.warning("No training reference files found")
            logger.info("Available files:")
            for obj in response['Contents']:
                logger.info(f"  - {obj['Key']}")
            return None
        
        # Retourne le run_id le plus récent
        latest_run_id = sorted(run_ids, key=lambda x: x[1], reverse=True)[0][0]
        logger.info(f"Latest run_id selected: {latest_run_id}")
        return latest_run_id
        
    except Exception as e:
        logger.error(f"Error getting latest run_id: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def load_reference_data(bucket: str, symbol: str, run_id: str | None = None) -> pd.DataFrame | None:
    """Charge les données de référence (train + validation) depuis S3"""
    try:
        # Si pas de run_id fourni, trouve le plus récent
        if run_id is None:
            run_id = get_latest_run_id(bucket, symbol)
            if run_id is None:
                return None
        
        logger.info(f"Loading reference data for run_id: {run_id}")
        
        # Charge les données d'entraînement et de validation
        train_key = f"{CONFIG['aws']['s3_model_prefix']}/reference/train_{run_id}.csv"
        val_key = f"{CONFIG['aws']['s3_model_prefix']}/reference/val_{run_id}.csv"
        
        logger.info(f"Trying to load train data from: s3://{bucket}/{train_key}")
        logger.info(f"Trying to load val data from: s3://{bucket}/{val_key}")
        
        train_df = load_csv_from_s3(bucket, train_key)
        val_df = load_csv_from_s3(bucket, val_key)
        
        if train_df is None and val_df is None:
            logger.warning(f"No reference data found for run_id: {run_id}")
            # Liste ce qui est disponible dans le répertoire reference
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=f"{CONFIG['aws']['s3_model_prefix']}/reference/"
                )
                if 'Contents' in response:
                    logger.info("Available files in reference directory:")
                    for obj in response['Contents']:
                        logger.info(f"  - {obj['Key']} (size: {obj['Size']}, modified: {obj['LastModified']})")
                else:
                    logger.warning("No files found in reference directory")
            except Exception as e:
                logger.error(f"Error listing reference directory: {e}")
            return None
        
        # Combine train et validation si les deux existent
        reference_dfs = []
        if train_df is not None:
            logger.info(f"Loaded training reference data: {len(train_df)} rows, columns: {list(train_df.columns)}")
            logger.info(f"Train data sample:\n{train_df.head()}")
            reference_dfs.append(train_df)
        
        if val_df is not None:
            logger.info(f"Loaded validation reference data: {len(val_df)} rows, columns: {list(val_df.columns)}")
            logger.info(f"Val data sample:\n{val_df.head()}")
            reference_dfs.append(val_df)
        
        if reference_dfs:
            reference_df = pd.concat(reference_dfs, ignore_index=True)
            logger.info(f"Combined reference data: {len(reference_df)} rows, columns: {list(reference_df.columns)}")
            logger.info(f"Combined data sample:\n{reference_df.head()}")
            return reference_df
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def load_reference_predictions(bucket: str, symbol: str, run_id: str | None = None) -> pd.DataFrame | None:
    """Charge les prédictions de référence (train + validation) depuis S3"""
    try:
        # Si pas de run_id fourni, trouve le plus récent
        if run_id is None:
            run_id = get_latest_run_id(bucket, symbol)
            if run_id is None:
                return None
        
        logger.info(f"Loading reference predictions for run_id: {run_id}")
        
        # Charge les prédictions d'entraînement et de validation
        train_pred_key = f"{CONFIG['aws']['s3_model_prefix']}/reference/train_predictions_{run_id}.csv"
        val_pred_key = f"{CONFIG['aws']['s3_model_prefix']}/reference/val_predictions_{run_id}.csv"
        
        logger.info(f"Trying to load train predictions from: s3://{bucket}/{train_pred_key}")
        logger.info(f"Trying to load val predictions from: s3://{bucket}/{val_pred_key}")
        
        train_pred_df = load_csv_from_s3(bucket, train_pred_key)
        val_pred_df = load_csv_from_s3(bucket, val_pred_key)
        
        if train_pred_df is None and val_pred_df is None:
            logger.warning(f"No reference predictions found for run_id: {run_id}")
            return None
        
        # Combine train et validation predictions si les deux existent
        reference_pred_dfs = []
        if train_pred_df is not None:
            logger.info(f"Loaded training predictions: {len(train_pred_df)} rows, columns: {list(train_pred_df.columns)}")
            reference_pred_dfs.append(train_pred_df)
        
        if val_pred_df is not None:
            logger.info(f"Loaded validation predictions: {len(val_pred_df)} rows, columns: {list(val_pred_df.columns)}")
            reference_pred_dfs.append(val_pred_df)
        
        if reference_pred_dfs:
            reference_pred_df = pd.concat(reference_pred_dfs, ignore_index=True)
            logger.info(f"Combined reference predictions: {len(reference_pred_df)} rows, columns: {list(reference_pred_df.columns)}")
            return reference_pred_df
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading reference predictions: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def load_reference_metrics(bucket: str, symbol: str) -> dict | None:
    """Charge les métriques de référence depuis S3"""
    metrics_key = f"{CONFIG['aws']['s3_model_prefix']}/metrics/metrics.json"
    logger.info(f"Loading reference metrics from s3://{bucket}/{metrics_key}")

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=metrics_key)
        body = obj['Body'].read()
        metrics_dict = json.loads(body)
        logger.info(f"Loaded reference metrics: {metrics_dict}")
        return metrics_dict
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.warning(f"Reference metrics not found: s3://{bucket}/{metrics_key}")
            return None
        else:
            logger.error(f"Error loading reference metrics: {e}")
            raise

def generate_evidently_report(current_df: pd.DataFrame, reference_predictions_df: pd.DataFrame, report_path: str) -> None:
    """Génère un rapport Evidently de performance en comparant les métriques actuelles vs référence"""
    if not EVIDENTLY_AVAILABLE:
        logger.info("Evidently package not available, skipping report generation")
        return

    if len(current_df) < 2:
        logger.warning("Not enough current data to generate Evidently report")
        return
    
    if reference_predictions_df is None or len(reference_predictions_df) < 2:
        logger.warning("Not enough reference prediction data to generate Evidently report")
        return

    try:
        # Prépare les données pour Evidently
        # Données actuelles : prediction, target, timestamp (données de production)
        curr_clean = current_df.drop(columns=["timestamp"], errors="ignore").copy()
        
        # Données de référence : prediction, target (données d'entraînement/validation)
        ref_clean = reference_predictions_df.copy()
        
        logger.info(f"Current data columns: {list(curr_clean.columns)}")
        logger.info(f"Reference predictions columns: {list(ref_clean.columns)}")
        
        # Vérification des colonnes nécessaires
        if 'target' not in curr_clean.columns or 'prediction' not in curr_clean.columns:
            logger.warning("Current data missing 'target' or 'prediction' columns")
            return
        
        if 'target' not in ref_clean.columns or 'prediction' not in ref_clean.columns:
            logger.warning("Reference predictions missing 'target' or 'prediction' columns")
            return
        
        # Sélectionne seulement les colonnes nécessaires
        current_for_comparison = curr_clean[['target', 'prediction']].copy()
        reference_for_comparison = ref_clean[['target', 'prediction']].copy()
        
        # Calcul des métriques pour logging
        ref_mae = mean_absolute_error(reference_for_comparison['target'], reference_for_comparison['prediction'])
        current_mae = mean_absolute_error(current_for_comparison['target'], current_for_comparison['prediction'])
        
        logger.info(f"Reference MAE (training baseline): {ref_mae:.4f}")
        logger.info(f"Current MAE (production): {current_mae:.4f}")
        logger.info(f"Performance degradation: {((current_mae - ref_mae) / ref_mae * 100):.2f}%")
        
        logger.info(f"Generating Evidently performance report with {len(reference_for_comparison)} reference rows and {len(current_for_comparison)} current rows")
        logger.info(f"Reference data sample (training baseline):\n{reference_for_comparison.head()}")
        logger.info(f"Current data sample (production):\n{current_for_comparison.head()}")
        
        # Utilise RegressionPreset pour comparer les performances de prédiction
        from evidently.metric_preset import RegressionPreset
        
        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=reference_for_comparison, current_data=current_for_comparison)
        report.save_html(report_path)
        logger.info(f"Evidently performance drift report saved at {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating Evidently report: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

def check_drift_and_trigger(**context):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours_back = 7 * 24  # 7 jours, heure par heure

    data_frames = []

    # Collecte les données des 7 derniers jours
    for hour_offset in range(hours_back):
        dt = now - timedelta(hours=hour_offset)
        date_str = dt.strftime("%Y-%m-%d")
        hour_str = dt.strftime("%H")

        pred_key = f"predictions/date={date_str}/hour={hour_str}/prediction.parquet"
        target_key = f"targets/date={date_str}/hour={hour_str}/target.parquet"

        pred_df = load_parquet(BUCKET_NAME, pred_key)
        target_df = load_parquet(BUCKET_NAME, target_key)

        if pred_df is not None and target_df is not None:
            prediction = pred_df.iloc[0, 0]
            target = target_df.iloc[0, 0]

            merged_df = pd.DataFrame({
                "prediction": [prediction],
                "target": [target],
                "timestamp": [dt]
            })

            data_frames.append(merged_df)
        else:
            logger.debug(f"Skipping hour {dt} due to missing prediction or target data")

    if not data_frames:
        logger.warning("No data loaded for the past 7 days; skipping drift check")
        return

    full_df = pd.concat(data_frames).sort_values("timestamp")
    logger.info(f"Loaded {len(full_df)} data points for drift analysis")

    # Calcule les MAE par fenêtre glissante
    mae_values = []
    timestamps = []

    # Ajuste le seuil minimum selon les données disponibles
    min_points_required = min(3, len(full_df) // 2)  # Au moins 3 points ou la moitié des données disponibles
    logger.info(f"Minimum points required per window: {min_points_required}")

    for hour_offset in range(min(24, hours_back)):  # Limite à 24 fenêtres pour éviter trop de calculs
        window_end = now - timedelta(hours=hour_offset)
        window_start = window_end - timedelta(hours=168)  # 7 jours

        window_df = full_df[(full_df["timestamp"] >= window_start) & (full_df["timestamp"] <= window_end)]
        if len(window_df) < min_points_required:
            logger.debug(f"Window {hour_offset}: only {len(window_df)} points, skipping")
            continue

        mae = mean_absolute_error(window_df["target"], window_df["prediction"])
        mae_values.append(mae)
        timestamps.append(window_end)
        logger.info(f"Window {hour_offset}: MAE = {mae:.4f} with {len(window_df)} points")

    if not mae_values:
        logger.warning("No MAE calculated with windowing approach")
        # Fallback: calcule le MAE sur toutes les données disponibles
        if len(full_df) >= 2:
            overall_mae = mean_absolute_error(full_df["target"], full_df["prediction"])
            mae_values = [overall_mae]
            timestamps = [now]
            logger.info(f"Using overall MAE: {overall_mae:.4f} with {len(full_df)} total points")
        else:
            logger.warning("Not enough data points for any MAE calculation; skipping drift check")
            return

    current_mae = mae_values[0] if mae_values else None
    mean_mae = sum(mae_values) / len(mae_values)
    
    logger.info(f"Current MAE: {current_mae:.4f}")
    logger.info(f"Mean MAE over analysis period: {mean_mae:.4f}")
    logger.info(f"Drift threshold: {DRIFT_THRESHOLD}")

    # Charge les métriques de référence pour comparaison
    reference_metrics = load_reference_metrics(BUCKET_NAME, SYMBOL)
    reference_mae = None
    if reference_metrics:
        reference_mae = reference_metrics.get('val_mae', reference_metrics.get('train_mae'))
        if reference_mae:
            logger.info(f"Reference MAE (from training): {reference_mae:.4f}")
            mae_degradation = (current_mae - reference_mae) / reference_mae if reference_mae > 0 else 0
            logger.info(f"MAE degradation vs training baseline: {mae_degradation:.2%}")
            
            # Comparaison avec un seuil de dégradation acceptable (ex: +20%)
            degradation_threshold = 0.20  # 20% de dégradation acceptable
            if mae_degradation > degradation_threshold:
                logger.warning(f"Significant performance degradation detected: {mae_degradation:.2%} > {degradation_threshold:.2%}")
        else:
            logger.warning("No reference MAE found in training metrics")
    else:
        logger.warning("No reference metrics available for comparison")

    # Génère le rapport Evidently
    report_file = os.path.join(REPORT_DIR, f"drift_report_{now.strftime('%Y%m%d%H%M')}.html")
    
    # Charge les prédictions de référence pour Evidently
    reference_predictions_df = load_reference_predictions(BUCKET_NAME, SYMBOL)
    if reference_predictions_df is not None:
        # Prépare les données actuelles dans le même format que les prédictions de référence
        current_for_evidently = full_df.copy()
        
        logger.info(f"Current data for Evidently: {list(current_for_evidently.columns)}")
        logger.info(f"Reference predictions columns: {list(reference_predictions_df.columns)}")
        
        generate_evidently_report(current_for_evidently, reference_predictions_df, report_file)
    else:
        logger.warning("Could not load reference predictions for Evidently report")

    # Décision de drift
    if current_mae and current_mae > DRIFT_THRESHOLD:
        logger.warning(f"Model drift detected! Current MAE ({current_mae:.4f}) > threshold ({DRIFT_THRESHOLD})")
        logger.warning("Triggering training DAG.")
        
        trigger_dag(
            dag_id="model_training_dag",
            run_id=f"drift-triggered-{now.strftime('%Y%m%d%H%M')}",
            conf={
                "reason": "drift_detected", 
                "current_mae": current_mae,
                "mean_mae": mean_mae,
                "threshold": DRIFT_THRESHOLD
            },
            execution_date=None,
            replace_microseconds=False,
        )
    else:
        logger.info("No significant drift detected.")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="drift_monitoring_dag",
    description="Hourly monitoring of model drift with Evidently performance report generation",
    schedule_interval="0 * * * *",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args=default_args,
) as dag:

    check_drift_task = PythonOperator(
        task_id="check_drift_and_trigger",
        python_callable=check_drift_and_trigger,
        provide_context=True,
    )

    check_drift_task