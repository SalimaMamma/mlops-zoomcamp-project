# ==========================================
# 📌 Makefile for Crypto Time Series MLOps
# ==========================================

# Colors for pretty output
GREEN := \033[0;32m
NC := \033[0m

.PHONY: help start stop logs init init-airflow init-mlflow init-localstack run-client test lint health format clean rebuild

help:
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  make start           - Build & start all Docker services"
	@echo "  make stop            - Stop all running Docker containers"
	@echo "  make logs            - Tail logs for all services"
	@echo "  make init            - Initialize Airflow, MLflow, LocalStack buckets"
	@echo "  make init-airflow    - Initialize Airflow DB and user"
	@echo "  make init-mlflow     - Setup MLflow backend/store"
	@echo "  make init-localstack - Create S3 buckets in LocalStack"
	@echo "  make run-client      - Run the prediction client script"
	@echo "  make test            - Run unit and integration tests"
	@echo "  make lint            - Run black, isort, flake8"
	@echo "  make format          - Auto-format code with black and isort"
	@echo "  make health          - Call FastAPI health endpoint"
	@echo "  make clean           - Remove __pycache__ & pyc files"
	@echo "  make rebuild         - Stop, rebuild & restart all services"

start:
	docker-compose -f docker-compose.airflow.yml up -d
	docker-compose -f docker-compose.api.yml up -d
stop:
	docker-compose -f docker-compose.airflow.yml down
	docker-compose -f docker-compose.api.yml down

logs:
	docker-compose -f docker-compose.airflow.yml logs -f
	docker-compose -f docker-compose.api.yml logs -f

init: 
	init-airflow init-mlflow init-localstack

init-airflow:
	docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow db init
	docker-compose -f docker-compose.airflow.yml exec airflow-webserver airflow users create \
	--username admin \
	--firstname admin \
	--lastname user \
	--role Admin \
	--email admin@example.com \
	--password admin

init-mlflow:
	@echo "✅ MLflow initialized (DB migrations if needed)."

init-localstack:
	awslocal s3 mb s3://crypto-pipeline-data
	awslocal s3 mb s3://crypto-pipeline-models

run-client:
	python src/client.py

test:
	pytest tests/

lint:
	black .
	isort .
	flake8 .

format:
	black .
	isort .

health:
	curl http://localhost:8000/health

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

rebuild: 
	stop
	docker-compose -f docker-compose.airflow.yml build
	docker-compose -f docker-compose.api.yml build
	make start
