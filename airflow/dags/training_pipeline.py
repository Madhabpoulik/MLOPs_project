from __future__ import annotations
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipelines.training_pipeline import TrainPipeline
import numpy as np

training_pipeline = TrainPipeline()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries": 2},
    description="This is the training pipeline",
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2024, 8, 18, tz="UTC"),
    catchup=False,
    tags=["training"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path,test_data_path = training_pipeline.start_data_ingestion()
        ti.xcom_push("ingested_file_path", {"train_data_path": train_data_path, "test_data_path": test_data_path})
        
    def data_transformation(**kwargs):
        ti = kwargs["ti"]
        ingested_file_path = ti.xcom_pull(task_ids="data_ingestion", key="ingested_file_path")
        train_data_path,test_data_path = ingested_file_path["train_data_path"], ingested_file_path["test_data_path"]
        train_arr,test_arr = training_pipeline.start_data_transformation(train_data_path,test_data_path)
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push("transformed_data", {"train_arr": train_arr, "test_arr": test_arr})
        
    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        transformed_data = ti.xcom_pull(task_ids="data_transformation", key="transformed_data")
        train_arr = np.array(transformed_data["train_arr"])
        test_arr = np.array(transformed_data["test_arr"])
        training_pipeline.start_model_training(train_arr,test_arr)
    
    def model_evaluation(**kwargs):
        ti = kwargs["ti"]
        transformed_data = ti.xcom_pull(task_ids="data_transformation", key="transformed_data")
        test_arr = np.array(transformed_data["test_arr"])
        training_pipeline.start_model_evaluation(test_arr)
        
data_ingestion_task = PythonOperator(
    task_id="data_ingestion",
    python_callable=data_ingestion,
    dag=dag,
)
data_ingestion_task.doc_md = dedent(
    """
    #### Data Ingestion
    This task is responsible for ingesting the data from the source.
    """
)

data_transformation_task = PythonOperator(
    task_id="data_transformation",
    python_callable=data_transformation,
    dag=dag,
)
data_transformation_task.doc_md = dedent(
    """
    #### Data Transformation
    This task is responsible for transforming the data.
    """
)

model_trainer_task = PythonOperator(
    task_id="model_trainer",
    python_callable=model_trainer,
    dag=dag,
)
model_trainer_task.doc_md = dedent(
    """
    #### Model Trainer
    This task is responsible for training the model.
    """
)

model_evaluation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=model_evaluation,
    dag=dag,
)
model_evaluation_task.doc_md = dedent(
    """
    #### Model Evaluation
    data_ingestion_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task
    """
)
    
data_ingestion_task >> data_transformation_task >> model_trainer_task >> model_evaluation_task 