import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os, sys
from dataclasses import dataclass
from pathlib import Path

from src.util.utils import save_object, evaluate_model

@dataclass
class ModelEvalutaionConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "notebooks", "data", "raw.csv")
        
    def initiate_model_evaluation(self):
        logging.info("Model evalutaion started")
        try:
            df = pd.read_csv(self.data_path)
            logging.info("Model evalutaion completed")
            return df
        except Exception as e:
            logging.info("Exception occured at Model evalutaion stage")
            raise customException(e, sys)


