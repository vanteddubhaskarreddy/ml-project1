import os
import sys

from src.exceptions import CustomException
from src.logger import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
# @dataclass: This is a decorator in Python that is used to automatically add special methods to classes which use mutable types. When you use @dataclass, Python automatically adds special methods like __init__ and __repr__ to your class. This means you don't have to write them yourself. In simple terms, it's a shortcut that makes your code cleaner and easier to read.
    
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Data ingestion successful")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # os.makedirs: This method is used to create a directory recursively. That means while creating a directory, if any intermediate-level directory is missing, os.makedirs() method will create them all and exists_ok parameter is used to avoid the error if the directory already exists.
            # os.path.dirname: This method in Python is used to get the directory name from the specified path. This method is used to split the path name into a pair of head and tail part. The tail part will be everything after the final slash.
            # self.ingestion_config is an object of the DataIngestionConfig class. It is used to access the attributes of the DataIngestionConfig class like train_data_path, test_data_path, and raw_data_path.
            # The index parameter is used to write row names. If True, the index of the DataFrame is written as the first column. If False, no index is written.
            # The header parameter is used to write the column names. If True, the column names are written. If False, they are not written. If you want to write the column names, you can set the header parameter to True.

            logging.info("Raw Data saved at %s", self.ingestion_config.raw_data_path)

            logging.info("Splitting data into train and test")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Train and Test Data split successful and ingested at %s and %s", self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
            logging.info("Data ingestion process completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))