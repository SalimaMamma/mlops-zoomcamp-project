#!/usr/bin/env bash
set -ex


until pg_isready -h postgres -U airflow; do
  echo "Waiting for postgres..."
  sleep 5
done


if [ "$1" = "webserver" ]; then
  echo "🗄️ Initialize Airflow"
  
  # Clear previous DB state
  #airflow db reset --yes
  airflow db init
  
  # Créer le fichier marqueur
  touch /opt/airflow/airflow_initialized
  
  : "${AIRFLOW_FIRSTNAME:=Admin}"
  : "${AIRFLOW_LASTNAME:=User}"
  : "${AIRFLOW_ROLE:=Admin}"

  echo "👤 Create user admin"
  airflow users create \
      --username "${AIRFLOW_USERNAME}" \
      --password "${AIRFLOW_PASSWORD}" \
      --firstname "${AIRFLOW_FIRSTNAME}" \
      --lastname "${AIRFLOW_LASTNAME}" \
      --role "${AIRFLOW_ROLE}" \
      --email "${AIRFLOW_EMAIL}" || echo "Utilisateur déjà existant"

  echo "🚀 Starting Airflow webserver..."
  exec airflow webserver

elif [ "$1" = "scheduler" ]; then
  # Le scheduler attend que le webserver ait initialisé la DB
  echo "⏳ Waiting for Airflow webserver to initialize the database..."
  while [ ! -f /opt/airflow/airflow_initialized ]; do
    echo "Waiting for database initialization..."
    sleep 5
  done
  
  echo "🚀 Starting Airflow scheduler..."
  exec airflow scheduler
else
  exec "$@"
fi