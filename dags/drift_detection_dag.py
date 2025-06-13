from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

path = os.getenv("Users/pranavr/drift-retrainer")

# Default args for both DAGs
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="drift_detection",
    default_args=default_args,
    description="Run drift_check.py on a schedule",
    schedule_interval="@hourly",          
    start_date=datetime(2025, 6, 10),
    catchup=False,
) as dag:

    drift_check = BashOperator(
        task_id="run_drift_check",
        bash_command="python src/drift_check.py",
        cwd=path,      # use pwd to find path
    )