#!/bin/sh
set -e

echo "✅ Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do sleep 1; done

echo "✅ Waiting for LocalStack..."
while ! nc -z localstack 4566; do sleep 1; done

echo "✅ Creating MLflow DB if not exists..."
PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -c "SELECT 1 FROM pg_database WHERE datname = 'mlflow';" | grep -q 1 \
  || PGPASSWORD=airflow psql -h postgres -U airflow -d airflow -c "CREATE DATABASE mlflow;"

echo "✅ Creating S3 bucket if not exists..."
aws --endpoint-url=http://localstack:4566 s3 mb s3://mlflow-bucket || true

echo "🚀 Starting MLflow..."
exec mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql+psycopg2://airflow:airflow@postgres:5432/mlflow \
  --default-artifact-root s3://mlflow-bucket/ \
  --workers 1 --gunicorn-opts "--timeout 120"
