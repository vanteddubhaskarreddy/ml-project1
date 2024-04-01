import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.modeltrainerconfig =  ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating model training, splitting data into train and test sets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            """{
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'AdaBoost Classifier': AdaBoostClassifier(),
                'XGB Classifier': XGBClassifier(),
                'CatBoosting Classifier': CatBoostClassifier(),
                'K Neighbours Classifier': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression()
            }"""

            model_report:dict = evaluate_models(X_train = X_train, X_test = X_test, y_train = y_train
                                                , y_test = y_test, models = models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.error("Best model score is less than 0.6, model training failed")
                raise CustomException("Best model score is less than 0.6, model training failed")
            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.modeltrainerconfig.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            logging.info(f"Model training successful with R2 score: {r2}")

            return r2
        except Exception as e:
            raise CustomException(e, sys)