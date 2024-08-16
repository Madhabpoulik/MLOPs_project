import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os, sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv(Path(os.path.join("data", "train.csv")))
            logging.info("Data read as pandas dataframe")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Data saved to raw data path")
            
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into train and test")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data split into train and test completed and saved")
            
            logging.info("Data Ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise customException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()