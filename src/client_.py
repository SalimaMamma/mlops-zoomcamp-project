import requests

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


if __name__ == "__main__":
    call_health()
    call_ingest()

    date = "2025-03-28"
    hour = "09"

    #call_preprocess(date, hour)
    call_predict(date, hour)
