from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("path-to-project")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_pipeline",
    default_args=default_args,
    description="Watch for drift flag and retrain model",
    schedule_interval="@hourly",            # same cadence as drift detection
    start_date=datetime(2025, 6, 10),
    catchup=False,
) as dag:

    wait_for_flag = FileSensor(
        task_id="wait_for_drift_flag",
        filepath="trigger_retrain.flag",
        fs_conn_id="fs_default",
        poke_interval=60,                 
        timeout=60 * 60,                   # 1 hour timeout
        mode="poke",
        dag=dag,
    )

    retrain = BashOperator(
        task_id="run_retrain_pipeline",
        bash_command="python src/retrain_pipeline.py",
        cwd=path,       # use pwd to find path
    )

    wait_for_flag >> retrain
