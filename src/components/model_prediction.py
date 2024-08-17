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
        
    def predict(self, input_data, model, preprocessor):
        try:
            data_scaled = preprocessor.transform(input_data)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured at Model evalutaion predict stage")
            raise customException(e, sys)
        
    def initiate_model_evaluation(self, actual_data):
        
        try:
            logging.info("Model evalutaion started")
            
            model = load_object(file_path=self.model_evaluation_config.model_path)
            preprocessor = load_object(file_path=self.model_evaluation_config.preprocessor_path)
            
            mlflow.set_registry_uri("https://dagshub.com/Madhabpoulik/MLOPs_project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"tracking_url_type_store : {tracking_url_type_store}")
            
            with mlflow.start_run():
                
                predicted_data = self.predict(actual_data, model, preprocessor)
                pred_datadf = pd.DataFrame(predicted_data)
                
                # rmse, mae, r2_square = self.eval_metrics(actual_data, predicted_data)
                
                # mlflow.log_metric("rmse", rmse)
                # mlflow.log_metric("mae", mae)
                # mlflow.log_metric("r2", r2_square)
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="Model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            print("top 5 predicted data", pred_datadf.head(5))
            logging.info("Model evalutaion completed")
            
            return predicted_data
        except Exception as e:
            logging.info("Exception occured at Model evalutaion stage")
            raise customException(e, sys)



if __name__ == "__main__":
    obj = ModelEvaluation()
    actual_data = pd.read_csv(obj.model_evaluation_config.test_data_path)
    print(actual_data.head(5))
    drop_columns = ["id"]
    input_feature_df = actual_data.drop(columns=drop_columns, axis=1)
    obj.initiate_model_evaluation(input_feature_df)