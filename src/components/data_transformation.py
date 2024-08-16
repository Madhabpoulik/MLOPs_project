import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customException

import os, sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.util.utils import save_object

@dataclass
class DataTransformationConfig:
    preproccessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_data_transformation(self):
        
        try:
            logging.info("Data transformation started")
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Pipeline initiated")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            logging.info("Pipeline completed")

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured at Data transformation stage")
            raise customException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data read as pandas dataframe")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = "price"
            drop_columns = [target_column_name, "id"]

            logging.info("segregating input and target feature for training data")
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            logging.info("segregating input and target feature for validation data")
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("tranforming input feature for training and validation data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            
            logging.info("Concatenating input and target feature for training and validation data")
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]
            
            save_object(
                file_path=self.data_tranformation_config.preproccessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured at Data transformation stage")
            raise customException(e, sys)


