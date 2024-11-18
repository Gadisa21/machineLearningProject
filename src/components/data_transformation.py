import sys

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.transformation_config=DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function is responsible for returning the preprocessor object'''
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder())
                    
                ]
            )
            logging.info("numerical columns:{}".format(numerical_columns))
            logging.info("categorical columns:{}".format(categorical_columns))

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipline,numerical_columns),
                    ("cat_pipline",cat_pipline,categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function is responsible for initiating the data transformation'''
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test data loaded successfully")
            logging.info("Obtaining the preprocessor object")

            preprocessor_obj=self.get_data_transformer_object()

            target_column="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]


            logging.info("Applying the preprocessor object on the train and test data")

            input_feature_train_transformed=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_transformed=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Data Transformation completed successfully")

            train_arr=np.c_[input_feature_train_transformed,np.array(target_feature_train_df)
                            ]
            test_arr=np.c_[input_feature_test_transformed,np.array(target_feature_test_df)]

            

            save_object(self.transformation_config.preprocessor_obj_file_path,preprocessor_obj)
            
            logging.info("Saved the preprocessor object")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)