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

            # The models dictionary contains the names of the models as keys and the model objects as values.
            # Linear Regression: This is like trying to draw a straight line that best fits your data points. It's used when there's a clear linear relationship between your input variables and the output. For example, predicting a person's weight based on their height.
            # Lasso Regression: This is similar to linear regression but it can reduce the impact of less important features by making their coefficients zero. This is useful when you have many input features and you want to focus on the most important ones. For example, predicting house prices based on features like size, location, number of rooms, etc.
            # Ridge Regression: This is also similar to linear regression but it reduces the impact of less important features without making their coefficients exactly zero. It's used when you want to keep all features but reduce the impact of less important ones.
            # K-Neighbors Regressor: This is like asking your neighbors for advice and averaging their opinions. It's used when your data points are clustered together and similar data points can be found near each other. For example, predicting a person's political preference based on the preferences of their neighbors.
            # Decision Tree Regressor: This is like making a flowchart to make a decision. It's used when you can make a series of yes/no questions that lead to your output. For example, predicting whether a person will buy a product based on questions like "Is the price less than $50?" or "Is the product a book?".
            # Random Forest Regressor: This is like asking a crowd of people (each person representing a decision tree) and taking the average of their opinions. It's used when a single decision tree is not accurate enough and you want to improve accuracy by averaging multiple decision trees.
            # XGBRegressor (Extreme Gradient Boosting): This is like a competition where each competitor tries to correct the mistakes of the previous competitor. It's used when you want to iteratively improve your model by focusing on the data points that are hard to predict.
            # CatBoosting Regressor: This is similar to XGBRegressor but it's especially good when you have categorical input variables. For example, predicting a person's favorite brand based on their past purchases.
            # AdaBoost Regressor: This is also similar to XGBRegressor but it gives more weight to the data points that are hard to predict. It's used when you want to focus on the difficult data points.
            

            '''
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                # Decision Tree
                # criterion: This is the function used to measure the quality of a split in the decision tree. For example, 'squared_error' means that the split that minimizes the sum of squared errors (differences between actual and predicted values) is chosen.
                # max_features: This is the number of features to consider when looking for the best split. 'sqrt' means the square root of the total number of features, and 'log2' means the base-2 logarithm of the total number of features.
                'Random Forest': {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', 'None'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                # Random Forest
                # n_estimators: This is the number of trees in the forest. For example, if n_estimators is 8, then the random forest will consist of 8 different decision trees.
                'Gradient Boosting': {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [.6, .7, .75, .8, .85, .9],
                    # 'criteria': ['friedman_mse', 'squared_error'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                # Gradient Boosting
                # learning_rate: This is a parameter that affects how quickly the model learns. A smaller learning rate means the model learns slowly, requiring more training iterations but often resulting in a better model.
                # subsample: This is the fraction of samples to be used for fitting the individual base learners. For example, if subsample is 0.6, then each base learner (decision tree) is trained on 60% of the total training samples, chosen randomly.
                # n_estimators: Similar to the Random Forest, this is the number of stages of boosting, i.e., the number of individual models to sequentially train.
                'Linear regression': {},
                'K Neighbours Regressor': {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                },
                # K Neighbours Regressor
                # n_neighbors: This is the number of neighbors to use for prediction. For example, if n_neighbors is 5, the model will look at the 5 closest data points to a given point to predict its value.
                # weights: This determines how much importance or "weight" is given to each neighbor in prediction. 'uniform' means all neighbors have equal weight, while 'distance' means closer neighbors have more weight.
                # algorithm: This is the algorithm used to compute the nearest neighbors. 'ball_tree', 'kd_tree', and 'brute' are different methods of calculating which data points are "neighbors".
                'XGBRegressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                # XGBRegressor
                # learning_rate: This is a parameter that affects how quickly the model learns. A smaller learning rate means the model learns slowly, requiring more training iterations but often resulting in a better model.
                # n_estimators: This is the number of stages of boosting, i.e., the number of individual models to sequentially train.
                'CatBoosting Regressor': {
                    'depth' : [6, 8 ,10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100],
                },
                # CatBoosting Regressor
                # depth: This is the maximum depth of the trees in the model. A tree of depth 6 will have up to 6 layers of decisions.
                # learning_rate: Similar to XGBRegressor, this affects how quickly the model learns.
                # iterations: This is the number of trees to be built, similar to n_estimators in the previous models.
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                }
                # AdaBoost Regressor
                # learning_rate: Again, this affects how quickly the model learns.
                # loss: This is the loss function to use when updating the weights. 'linear', 'square', and 'exponential' are different ways of calculating the difference between the predicted and actual values.
                # n_estimators: This is the maximum number of estimators at which boosting is terminated. In other words, it's the maximum number of models to train sequentially.
            }
            '''
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train = X_train, X_test = X_test, y_train = y_train
                                                , y_test = y_test, models = models, params = params)
            
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