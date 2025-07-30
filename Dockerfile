FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src
# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

COPY wait-for-bucket.sh /wait-for-bucket.sh
RUN chmod +x /wait-for-bucket.sh

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
