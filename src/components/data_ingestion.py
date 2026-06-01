import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = (
            DataIngestionConfig()
        )

    def initiate_data_ingestion(self):

        logging.info(
            "Entered Data Ingestion Method"
        )

        try:

            # Read Dataset
            df = pd.read_csv(
                r"notebook\data\Telco Customer Churn.csv"
            )

            logging.info(
                "Dataset loaded successfully"
            )

            # Drop Customer ID
            if "customerID" in df.columns:
                df.drop(
                    "customerID",
                    axis=1,
                    inplace=True
                )

            # Convert TotalCharges
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

            # Fill Missing Values
            df["TotalCharges"] = (
                df["TotalCharges"]
                .fillna(
                    df["TotalCharges"].median()
                )
            )

            logging.info(
                "Data cleaning completed"
            )

            # Create Artifacts Folder
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save Raw Data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw data saved"
            )

            # Train Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Churn"]
            )

            # Save Train Data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save Test Data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Train Test Split Completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_data_path, test_data_path = (
        obj.initiate_data_ingestion()
    )

    data_transformation = (
        DataTransformation()
    )

    train_arr, test_arr, _ = (
        data_transformation
        .initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )

    model_trainer = ModelTrainer()

    print(
        model_trainer
        .initiate_model_trainer(
            train_arr,
            test_arr
        )
    )