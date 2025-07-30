import logging
from datetime import datetime
import boto3
import mlflow
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Import des fonctions métier
from src.inference import make_prediction, run_preprocess, ingest_data
from src.data_processing import CryptoDataProcessor

# --- Configuration ---
with open("src/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# --- Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi-crypto-api")

# --- S3 Client ---
s3_client = boto3.client(
    "s3",
    endpoint_url=CONFIG["aws"]["endpoint_url"],
    aws_access_key_id=CONFIG["aws"]["access_key_id"],
    aws_secret_access_key=CONFIG["aws"]["secret_access_key"],
    region_name="us-east-1",
)

BUCKET_NAME = CONFIG["aws"]["s3_bucket"]
SYMBOL = CONFIG["data"]["symbols"][0]

# --- MLflow setup ---
MODEL_NAME = f"{CONFIG['mlflow']['model_name']}_{SYMBOL.replace('-', '_')}"
mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])

try:
    mlflow_client = mlflow.tracking.MlflowClient()
    rm = mlflow_client.search_registered_models(f"name='{MODEL_NAME}'")[0]
    prod_version_number = rm.aliases.get("production")
    if not prod_version_number:
        raise Exception(f"No 'production' alias for {MODEL_NAME}")

    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{prod_version_number}")
    logging.info(f"✅ Model loaded: {MODEL_NAME}, version {prod_version_number}")
except Exception as e:
    model = None
    logger.error(f"MLflow client init failed: {e}")

processor = CryptoDataProcessor(CONFIG)

# --- Pydantic models ---
class HealthResponse(BaseModel):
    status: str = Field("ok", example="ok")
    timestamp: datetime = Field(default_factory=datetime.utcnow, example="2025-07-28T15:32:00Z")

class IngestResponse(BaseModel):
    message: str = Field(..., example="Ingestion finished")
    rows_fetched: int = Field(..., ge=0, example=1000)

class PreprocessResponse(BaseModel):
    message: str = Field(..., example="Preprocessing done")
    rows_processed: int = Field(..., ge=0, example=900)

class PredictResponse(BaseModel):
    prediction: float = Field(..., example=0.0345)
    cached: bool = Field(..., example=False)

# --- FastAPI App ---
app = FastAPI(title="Crypto Prediction API")

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    result = ingest_data(CONFIG)
    return IngestResponse(**result)

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(date: str = Query(...), hour: int = Query(...)):
    try:
        date_dt = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Date must be YYYY-MM-DD format")
    
    rows = run_preprocess(date_dt, hour, processor, SYMBOL, s3_client, BUCKET_NAME)
    return PreprocessResponse(message="Preprocessing done", rows_processed=rows)

@app.get("/predict", response_model=PredictResponse)
def predict(
    date: str = Query(..., description="Date YYYY-MM-DD"),
    hour: int = Query(..., ge=0, le=23, description="Hour 0-23"),
):
    result = make_prediction(date, hour, CONFIG, s3_client, BUCKET_NAME, model, processor, SYMBOL)
    return PredictResponse(**result)