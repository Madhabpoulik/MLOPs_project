from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

import os, sys
from src.logger.logging import logging
from src.exception.exception import customException

class TrainPipeline:
    def __init__(self):
        pass
    
    def start_data_ingestion(self):
        try:
            logging.info("Data Ingestion stage started")
            data_ingestion = DataIngestion()
            train_data_path, validation_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, validation_data_path
        
        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise customException(e, sys)
    
    def start_data_transformation(self, train_data_path, validation_data_path):
        
        try:
            logging.info("Data Transformation stage started")
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, validation_data_path)
            return train_arr, test_arr
        
        except Exception as e:
            logging.info("Exception occured at Data Transformation stage")
            raise customException(e, sys)
        
    def start_model_training(self, train_arr, test_arr):
        
        try:
            logging.info("Model Training stage started")
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)

        except Exception as e:
            logging.info("Exception occured at Model Training stage")
            raise customException(e, sys)
        
    def start_model_evaluation(self, test_arr):

        try:
            logging.info("Model Evaluation stage started")
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(test_arr)
            
        except Exception as e:
            logging.info("Exception occured at Model Evaluation stage")
            raise customException(e, sys)
        
    def start_process(self):
        try:
            logging.info("Entered Training Pipeline")
            train_data_path, validation_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, validation_data_path)
            self.start_model_training(train_arr, test_arr)
            self.start_model_evaluation(test_arr)
        except Exception as e:
            logging.info("Exception occured at Training Pipeline stage")
            raise customException(e, sys)
        

if __name__=="__main__":
    obj = TrainPipeline()
    obj.start_process()