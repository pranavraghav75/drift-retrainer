from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

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
    schedule="@hourly",
    start_date=datetime(2025, 6, 10),
    catchup=False,
) as dag:

    run_drift_check = BashOperator(
        task_id = "run_drift_check",
        bash_command="python src/drift_check.py",
        cwd="/Users/pranavr/drift-retrainer",  # use pwd to find path
    )
