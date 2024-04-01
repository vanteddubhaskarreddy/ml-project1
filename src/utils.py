import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved at %s", file_path)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    logging.info("Evaluating models")
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models)[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score
            logging.info(f"Model: {model_name} \t Train Score: {train_score} \t Test Score: {test_score}")
        logging.info("Model evaluation successful")
        # logging.info(f"The model evaluation report is: {report}")
        return report
            
    except Exception as e:
        raise CustomException(e, sys)
    