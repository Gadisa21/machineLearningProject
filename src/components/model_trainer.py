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

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact", "trained_model.pkl")

class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours classifier":KNeighborsRegressor(),
                "XGBclassifier":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "AdaBoost classifier":AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]

            if best_model_score<0.6:
                logging.error("Model score is less than 0.6. No best model found.")
                raise CustomException("Model score is less than 0.6")

            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")

            save_object(file_path=self.config.trained_model_file_path,obj=best_model)
            predicted=best_model.predict(x_test)
            r_score=r2_score(y_test,predicted)

            return r_score

        except Exception as e:
            raise CustomException(e,sys)
            
        