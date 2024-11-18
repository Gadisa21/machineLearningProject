import os
import sys
from src.exception import CustomException   
from sklearn.model_selection import train_test_split
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    trian_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("read the dataset ad data frame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
        
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Training and Testing data split started")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.trian_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.trian_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    data_ingestion=DataIngestion()
    train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path,test_data_path) 