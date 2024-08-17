import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os, sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.util.utils import save_object, evaluate_model, load_object

from urllib.parse import urlparse
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

@dataclass
class ModelEvalutaionConfig:
    model_path:str = os.path.join("artifacts", "model.pkl")
    preprocessor_path:str = os.path.join("artifacts", "preprocessor.pkl")
    validation_data_path:str = os.path.join("artifacts", "validation.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvalutaionConfig()
    
    def eval_metrics(self, actual_data, pred_data):
        try:
            mae = mean_absolute_error(actual_data, pred_data)
            mse = mean_squared_error(actual_data, pred_data)
            rmse = np.sqrt(mse)
            r2_square = r2_score(actual_data, pred_data)
            return mae, rmse, r2_square
        except Exception as e:
            logging.info("Exception occured at Model evalutaion eval_metrics stage")
            raise customException(e, sys)
        
    def initiate_model_evaluation(self, test_arr):
        
        try:
            logging.info("Model evalutaion started")
            
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            
            model = load_object(file_path=self.model_evaluation_config.model_path)
            
            mlflow.set_registry_uri("https://dagshub.com/Madhabpoulik/MLOPs_project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"tracking_url_type_store : {tracking_url_type_store}")
            mlflow.set_experiment("MLOPs_project")
            
            with mlflow.start_run():
                
                predicted_data = model.predict(x_test)
                                
                rmse, mae, r2_square = self.eval_metrics(y_test, predicted_data)
                logging.info("logging metrics in ml flow")
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2_square)
                logging.info("mlflow logging completed")
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="Model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            logging.info("Model evalutaion completed")
            
        except Exception as e:
            logging.info("Exception occured at Model evalutaion stage")
            raise customException(e, sys)