from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipelines.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries": 2},
    description="This is the training pipeline",
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2024, 8, 18, tz="UTC"),
    catchuup=False,
    tags=["training"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        training_pipeline = TrainingPipeline()
        training_pipeline.start_data_ingestion()
        ti.xcom_push(key="ingested_file_path", value=training_pipeline.data_ingestion_config.ingested_dir)