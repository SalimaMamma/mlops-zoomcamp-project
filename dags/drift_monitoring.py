import io
import logging
import os
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from airflow import DAG
from airflow.api.common.experimental.trigger_dag import trigger_dag
from airflow.operators.python import PythonOperator
from botocore.exceptions import ClientError
from sklearn.metrics import (mean_absolute_error,  # Importé au début
                             mean_squared_error, r2_score)

# 🔹 Import Evidently avec gestion d'erreur pour compatibilité
try:
    from evidently.report import Report
    from evidently.metric_preset import RegressionPreset, DataDriftPreset
    from evidently.metrics import RegressionQualityMetric
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Could not import Evidently: {e}")
    EVIDENTLY_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("/opt/airflow/src/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

S3_ENDPOINT = "http://localstack:4566"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
BUCKET_NAME = CONFIG['aws']['s3_bucket']

SYMBOL = CONFIG['data']['symbols'][0]
DRIFT_THRESHOLD = CONFIG['model']['mae_drift_threshold']

# S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="us-east-1",
)

REPORT_DIR = "/opt/airflow/dags/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5)
}

def generate_evidently_report_safe(df_eval, report_path):
    try:
        if not EVIDENTLY_AVAILABLE:
            logger.warning("⚠️ Evidently not available, skipping report generation")
            return False

        df_report = df_eval.copy()

        if 'timestamp' in df_report.columns:
            df_report = df_report.drop('timestamp', axis=1)

        if 'target' not in df_report.columns or 'prediction' not in df_report.columns:
            logger.error("❌ Missing 'target' or 'prediction' columns for Evidently report")
            return False

        try:
            report = Report(metrics=[
                RegressionPreset(),
                DataDriftPreset()
            ])

            split_idx = len(df_report) // 2
            reference_data = df_report.iloc[:split_idx]
            current_data = df_report.iloc[split_idx:]

            if len(reference_data) == 0 or len(current_data) == 0:
                logger.warning("⚠️ Not enough data to split for Evidently report, using entire dataset for both")
                reference_data = df_report
                current_data = df_report

            report.run(reference_data=reference_data, current_data=current_data)
            report.save_html(report_path)
            logger.info(f"✅ Evidently report generated at: {report_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error generating Evidently report: {e}")
            logger.info("🔄 Trying simplified Evidently report generation")

            try:
                simple_report = Report(metrics=[RegressionQualityMetric()])
                simple_report.run(reference_data=df_report, current_data=df_report)
                simple_report.save_html(report_path)
                logger.info(f"✅ Simplified Evidently report generated at: {report_path}")
                return True
            except Exception as e2:
                logger.error(f"❌ Error generating simplified Evidently report: {e2}")
                return False

    except Exception as e:
        logger.error(f"❌ Unexpected error in Evidently report generation: {e}")
        return False

def generate_fallback_report(df_eval, mae_list, mean_mae, report_path):
    try:
        if 'target' not in df_eval.columns or 'prediction' not in df_eval.columns:
            logger.error("❌ Missing 'target' or 'prediction' columns in evaluation data")
            return

        mae_val = mean_absolute_error(df_eval['target'], df_eval['prediction'])
        mse_val = mean_squared_error(df_eval['target'], df_eval['prediction'])
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(df_eval['target'], df_eval['prediction'])

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Monitoring Report - Fallback</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                .data-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Drift Monitoring Report</h1>
                    <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Analyzed period:</strong> Rolling 7 days</p>
                </div>

                <div class="{'alert' if mean_mae > DRIFT_THRESHOLD else 'success'}">
                    <h3>{'🚨 DRIFT DETECTED!' if mean_mae > DRIFT_THRESHOLD else '✅ NO DRIFT DETECTED'}</h3>
                    <p><strong>Average MAE:</strong> {mean_mae:.4f}</p>
                    <p><strong>Configured threshold:</strong> {DRIFT_THRESHOLD:.4f}</p>
                    <p><strong>Number of hours analyzed:</strong> {len(mae_list)}</p>
                </div>

                <h2>📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{mae_val:.4f}</div>
                        <div class="metric-label">Mean Absolute Error (MAE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{mse_val:.4f}</div>
                        <div class="metric-label">Mean Squared Error (MSE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{rmse_val:.4f}</div>
                        <div class="metric-label">Root Mean Squared Error (RMSE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{r2_val:.4f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                </div>

                <h2>📈 MAE Evolution by Hour</h2>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Hour</th>
                            <th>MAE</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for i, mae in enumerate(mae_list):
            status = "🚨 Above threshold" if mae > DRIFT_THRESHOLD else "✅ Normal"
            html_content += f"""
                        <tr style="{'background-color: #ffebee;' if mae > DRIFT_THRESHOLD else ''}">
                            <td>Hour {i+1}</td>
                            <td>{mae:.4f}</td>
                            <td>{status}</td>
                        </tr>
            """

        html_content += f"""
                    </tbody>
                </table>

                <h2>ℹ️ Technical Information</h2>
                <ul>
                    <li><strong>Total number of samples:</strong> {len(df_eval)}</li>
                    <li><strong>Analysis period:</strong> Rolling 7 days</li>
                    <li><strong>Frequency:</strong> Weekly</li>
                    <li><strong>MAE drift threshold:</strong> {DRIFT_THRESHOLD:.4f}</li>
                </ul>

                <div class="alert">
                    <strong>Note:</strong> This report was generated in fallback mode because Evidently was not available or encountered a compatibility error.
                </div>
            </div>
        </body>
        </html>
        """

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✅ Fallback report generated: {report_path}")

    except Exception as e:
        logger.error(f"❌ Error generating fallback report: {e}")

def check_drift_and_report(**context):
    exec_date = datetime.now(timezone.utc)

    end_date = exec_date - timedelta(hours=1)
    start_date = end_date - timedelta(days=7)

    logger.info(f"Checking drift from {start_date} to {end_date}")

    mae_list = []
    dfs_eval = []

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        hour_str = current.strftime("%H")

        key_pred = f"predictions/date={date_str}/hour={hour_str}/prediction.parquet"
        key_target = f"targets/date={date_str}/hour={hour_str}/target.parquet"

        try:
            logger.info(f"📥 Downloading {key_pred} and {key_target}")

            obj_pred = s3_client.get_object(Bucket=BUCKET_NAME, Key=key_pred)
            obj_target = s3_client.get_object(Bucket=BUCKET_NAME, Key=key_target)

            buffer_pred = io.BytesIO(obj_pred['Body'].read())
            buffer_target = io.BytesIO(obj_target['Body'].read())

            df_pred = pq.read_table(buffer_pred).to_pandas()
            df_target = pq.read_table(buffer_target).to_pandas()

            if 'prediction' not in df_pred.columns or 'target' not in df_target.columns:
                logger.warning(f"⚠️ Missing columns in {date_str} {hour_str} → skipping")
                current += timedelta(hours=1)
                continue

            if len(df_pred) != len(df_target):
                logger.warning(f"⚠️ Size mismatch for {date_str} {hour_str} (predictions: {len(df_pred)}, targets: {len(df_target)}) → skipping")
                current += timedelta(hours=1)
                continue

            y_pred = df_pred['prediction'].values
            y_true = df_target['target'].values

            mae = np.mean(np.abs(y_true - y_pred))
            mae_list.append(mae)

            logger.info(f"✅ MAE for {date_str} {hour_str} = {mae:.4f}")

            df_eval = pd.DataFrame({
                'timestamp': current,
                'prediction': y_pred,
                'target': y_true
            })
            dfs_eval.append(df_eval)

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"⚠️ Missing file: {key_pred} or {key_target} → skipping")
            else:
                raise

        except Exception as e:
            logger.warning(f"⚠️ Unexpected error on {date_str} {hour_str}: {e}")

        current += timedelta(hours=1)

    if len(mae_list) == 0:
        logger.warning("❌ No data available for the period — cannot check drift.")
        return

    mean_mae = np.mean(mae_list)
    logger.info(f"📊 Average MAE over {len(mae_list)} hours: {mean_mae:.4f} | Threshold: {DRIFT_THRESHOLD:.4f}")

    if dfs_eval:
        df_eval_all = pd.concat(dfs_eval, ignore_index=True)
        report_path = os.path.join(REPORT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

        evidently_success = generate_evidently_report_safe(df_eval_all, report_path)

        if not evidently_success:
            logger.info("🔄 Generating fallback report...")
            generate_fallback_report(df_eval_all, mae_list, mean_mae, report_path)
    else:
        logger.warning("⚠️ Not enough data to generate a report")

    if mean_mae > DRIFT_THRESHOLD:
        logger.warning(f"🚨 Drift detected! Average MAE = {mean_mae:.4f} > threshold = {DRIFT_THRESHOLD:.4f}")
        try:
            trigger_dag(
                dag_id="training_pipeline",
                run_id=f"triggered_by_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                conf=None
            )
            logger.info("✅ Training pipeline triggered successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to trigger training DAG: {e}")
    else:
        logger.info("✅ No drift detected. Training will not be triggered.")

with DAG(
    "drift_monitoring",
    default_args=default_args,
    description="Drift monitoring DAG that triggers training when drift is detected",
    schedule_interval="0 * * * *",  # every hour at minute 0
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
) as dag:

    drift_monitoring_task = PythonOperator(
        task_id="check_drift_and_report",
        python_callable=check_drift_and_report,
        provide_context=True,
    )
