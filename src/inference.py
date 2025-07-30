import io
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
from fastapi import HTTPException

from src.data_ingestion import CoinbaseDataClient
from src.data_processing import CryptoDataProcessor

logger = logging.getLogger("crypto-services")

def run_preprocess(date_dt: date, hour: int, processor, symbol: str, s3_client, bucket_name: str, data=None) -> int:
    """
    Fonction interne pure Python pour le preprocessing - VOTRE CODE ORIGINAL
    """
    
    if data is None:
        data = processor.combine_datasets(symbol)

    df_features = processor.engineer_features(data)
    X, y = processor.prepare_features_target(df_features)

    remove_cols = ["index"]
    X = X.drop(columns=[c for c in remove_cols if c in X.columns], errors="ignore")
    hours_ftrs = hour - 1
    if hour < 0:
        date_dt = date_dt - timedelta(days=1)
        hours_ftrs = 23
    logger.info(f"Filtering features for date={date_dt} hour={hours_ftrs}")
    last_index = X.index.max()
    logger.info(
        "Last timestamp in features: "
        + str(X.loc[last_index]["day_of_month"])
        + "-"
        + ""
        + str(X.loc[last_index]["month"])
        + " "
        + str(str(X.loc[last_index]["hour"]))
    )
    mask = (
        (X["day_of_month"] == date_dt.day)
        & (X["month"] == date_dt.month)
        & (X["hour"] == hours_ftrs)
    )

    df_filtered = X.loc[mask]
    y = y[mask]

    processor.load_preprocessor(f"scaler_{symbol.replace('-', '_')}.pickle")
    X_scaled = processor.scaler.transform(df_filtered)
    df_scaled = pd.DataFrame(
        X_scaled, columns=df_filtered.columns, index=df_filtered.index
    )
    df_scaled["target"] = y

    # Save features and target to S3 as parquet
    table_features = pa.Table.from_pandas(df_scaled.reset_index())
    buffer_features = io.BytesIO()
    pq.write_table(table_features, buffer_features)
    buffer_features.seek(0)
    key_features = f"features/date={date_dt}/hour={hour}/features.parquet"
    s3_client.put_object(
        Bucket=bucket_name, Key=key_features, Body=buffer_features.getvalue()
    )

    df_target = df_scaled[["target"]].reset_index(drop=True)
    table_target = pa.Table.from_pandas(df_target)
    buffer_target = io.BytesIO()
    pq.write_table(table_target, buffer_target)
    buffer_target.seek(0)
    key_target = f"targets/date={date_dt}/hour={hour}/target.parquet"
    s3_client.put_object(
        Bucket=bucket_name, Key=key_target, Body=buffer_target.getvalue()
    )

    return len(df_filtered)


def ingest_data(config: dict) -> dict:
    """
    Fonction pour ingérer les données
    """
    try:
        client = CoinbaseDataClient(config)
        df = client.fetch_and_save_historical(days_back=1)  # 1 day back ingestion
        rows = len(df)
        logger.info(f"Ingestion done, rows fetched: {rows}")
        return {"message": "Ingestion finished", "rows_fetched": rows}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")


def make_prediction(date_str: str, hour: int, config: dict, s3_client, bucket_name: str, model, processor, symbol: str) -> dict:

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        date_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Date must be YYYY-MM-DD")

    # Conversion timezone 
    local_tz_str = "Europe/Paris"
    local_dt = datetime.combine(date_dt, datetime.min.time()).replace(hour=hour)
    local_tz = pytz.timezone(local_tz_str)
    local_dt = local_tz.localize(local_dt)

   
    utc_dt = local_dt.astimezone(pytz.utc)
    logger.info(f"UTC date: {utc_dt} (type={type(utc_dt)})")

   
    date = utc_dt.date()
    hour = utc_dt.hour
    
    logger.info(f"Processing date={date} hour={hour}")
    key_features = f"features/date={date}/hour={hour}/features.parquet"

    try:
   
        obj = s3_client.get_object(Bucket=bucket_name, Key=key_features)
        buffer = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buffer)
        df_features = table.to_pandas()
        logger.info(f"✅ Features loaded from S3 for date={date} hour={hour}")

    except s3_client.exceptions.NoSuchKey:
        logger.warning(
            f"⚠️ Features not found for date={date} hour={hour} → fallback triggered"
        )

       
        HISTORICAL_DAYS_BACK = 40
        try:
            client = CoinbaseDataClient(config)
            df_raw = client.fetch_and_save_historical(
                utc_dt, days_back=HISTORICAL_DAYS_BACK
            )
            logger.info(
                f"✅ Ingested {len(df_raw)} rows for fallback (last {HISTORICAL_DAYS_BACK} days)"
            )
        except Exception as e:
            logger.error(f"❌ Ingestion failed during fallback: {e}")
            raise HTTPException(status_code=500, detail="Fallback ingestion failed")

        # 4️⃣ Relancer le preprocess pour la date/hour 
        try:
            run_preprocess(date_dt=date, hour=hour, processor=processor, symbol=symbol, s3_client=s3_client, bucket_name=bucket_name, data=df_raw)
        except Exception as e:
            logger.error(f"❌ Preprocess failed during fallback: {e}")
            raise HTTPException(status_code=500, detail="Fallback preprocessing failed")

        # 5️⃣ Retenter de charger les features fraîchement créés 
        try:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key_features)
            buffer = io.BytesIO(obj["Body"].read())
            table = pq.read_table(buffer)
            df_features = table.to_pandas()
            logger.info(f"✅ Features created and loaded for date={date} hour={hour}")
        except Exception as e:
            logger.error(f"❌ Could not load fallback features: {e}")
            raise HTTPException(
                status_code=500, detail="Could not load fallback features"
            )

    except Exception as e:
        logger.error(f"❌ Unexpected error while loading features: {e}")
        raise HTTPException(status_code=500, detail="Feature loading failed")

    if df_features.empty:
        raise HTTPException(status_code=404, detail="Features file is empty")

    # 6️⃣ Reload scaler 
    scaler_filename = f"scaler_{symbol.replace('-', '_')}.pickle"
    processor.load_preprocessor(scaler_filename)

    # 7️⃣ Préparer les features 
    to_drop = ["timestamp", "symbol", "index", "target"]
    X_pred = df_features.drop(
        columns=[c for c in to_drop if c in df_features.columns], errors="ignore"
    )

    # 8️⃣ Scaling 
    X_scaled = processor.scaler.transform(X_pred)

    # 9️⃣ Predict 
    y_pred = model.predict(X_scaled)

    logger.info(f"✅ Prediction done for date={date} hour={hour} → y_pred={y_pred[0]}")

    return {"prediction": float(y_pred[0]), "cached": False}