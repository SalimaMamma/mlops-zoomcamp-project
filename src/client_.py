import requests
import argparse
from datetime import datetime

BASE_URL = "http://localhost:8000"

def call_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.status_code, r.json())

def call_ingest():
    r = requests.post(f"{BASE_URL}/ingest")
    print("Ingest:", r.status_code, r.json())

def call_preprocess(date: str, hour: int):
    params = {"date": date, "hour": hour}
    r = requests.post(f"{BASE_URL}/preprocess", params=params)
    print("Preprocess:", r.status_code, r.json())

def call_predict(date: str, hour: int):
    params = {"date": date, "hour": hour}
    r = requests.get(f"{BASE_URL}/predict", params=params)
    if r.status_code == 200:
        print("Prediction:", r.json())
    else:
        print("Error:", r.status_code, r.json())

def validate_date(date_str: str) -> bool:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_hour(hour_str: str) -> bool:
    """Validate hour is between 0-23"""
    try:
        hour = int(hour_str)
        return 0 <= hour <= 23
    except ValueError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call Crypto Prediction API")
    parser.add_argument("--date", type=str, required=True, 
                       help="Date in YYYY-MM-DD format (e.g., 2025-03-28)")
    parser.add_argument("--hour", type=str, required=True,
                       help="Hour from 0 to 23 (e.g., 9 or 09)")
    parser.add_argument("--action", type=str, choices=["health", "ingest", "preprocess", "predict", "all"],
                       default="predict", help="Action to perform (default: predict)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_date(args.date):
        print("❌ Error: Date must be in YYYY-MM-DD format")
        exit(1)
    
    if not validate_hour(args.hour):
        print("❌ Error: Hour must be between 0 and 23")
        exit(1)
    
    hour = int(args.hour)
    
    print(f"🚀 Calling API with date={args.date} and hour={hour}")
    print("-" * 50)
    
    if args.action == "health":
        call_health()
    elif args.action == "ingest":  
        call_ingest()
    elif args.action == "preprocess":
        call_preprocess(args.date, hour)
    elif args.action == "predict":
        call_predict(args.date, hour)
    elif args.action == "all":
        call_health()
        print()
        call_ingest()
        print()
        call_preprocess(args.date, hour)
        print()
        call_predict(args.date, hour)