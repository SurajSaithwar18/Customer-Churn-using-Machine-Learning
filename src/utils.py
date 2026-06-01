import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(
            file_path
        )

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        logging.info(
            f"Saving object at {file_path}"
        )

        with open(
            file_path,
            "wb"
        ) as file_obj:

            dill.dump(
                obj,
                file_obj
            )

        logging.info(
            "Object saved successfully"
        )

    except Exception as e:

        logging.error(
            "Error occurred while saving object"
        )

        raise CustomException(
            e,
            sys
        )


def load_object(file_path):

    try:

        logging.info(
            f"Loading object from {file_path}"
        )

        with open(
            file_path,
            "rb"
        ) as file_obj:

            obj = dill.load(
                file_obj
            )

        logging.info(
            "Object loaded successfully"
        )

        return obj

    except Exception as e:

        logging.error(
            "Error occurred while loading object"
        )

        raise CustomException(
            e,
            sys
        )