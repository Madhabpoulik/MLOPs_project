import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os, sys
from dataclasses import dataclass

from src.util.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor

divider = "="*40
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_arr, test_arr):
        
        try:
            logging.info("Model training started")
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "XGBRegressor": XGBRegressor()
            }
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print(f"\n{divider}\n")
            logging.info(f"Model Report : {model_report}")
            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print(f"Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}")
            print(f"\n{divider}\n")
            logging.info(f"Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training completed")
            
        except Exception as e:
            logging.info("Exception occured at Model training stage")
            raise customException(e, sys)


