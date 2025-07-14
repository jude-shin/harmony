import os
import logging 

from utils.product_lines import PRODUCTLINES as PLS


DATA_DIR_ENV= 'DATA_DIR'
SAVED_MODEL_DIR_ENV = 'SAVED_MODEL_DIR'

TRAIN_DATASET_PATH_ENV = 'TRAIN_DATASET_PATH_NAME'
VAL_DATASET_PATH_ENV = 'VAL_DATASET_PATH_NAME'
IMAGES_PATH_ENV = 'IMAGES_PATH_NAME'

def get_dir(n: str) -> str:
    d = os.getenv(n)
    if d is None:
        msg = n + ' env var is not set. Check the .env file.'
        logging.error(msg)
        raise KeyError(msg)

    return d

def get_data_dir() -> str:
    return get_dir(DATA_DIR_ENV)

def get_saved_model_dir() -> str:
    return get_dir(SAVED_MODEL_DIR_ENV)

def get_val_dataset_path(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, TRAIN_DATASET_PATH_ENV)

def get_train_dataset_path(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, VAL_DATASET_PATH_ENV)

def get_images_dir() -> str:
    return os.path.join(get_data_dir(), pl.value, IMAGES_PATH_ENV)

