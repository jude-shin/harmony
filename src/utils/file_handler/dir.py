import os
import logging 

from utils.product_lines import PRODUCTLINES as PLS

DATA_DIR_ENV= 'DATA_DIR'
SAVED_MODEL_DIR_ENV = 'SAVED_MODEL_DIR'

TRAIN_DATASET_PATH_ENV = 'TRAIN_DATASET_PATH_NAME'
VAL_DATASET_PATH_ENV = 'VAL_DATASET_PATH_NAME'
IMAGES_PATH_ENV = 'IMAGES_PATH_NAME'

def get_env(n: str) -> str:
    d = os.getenv(n)
    if d is None:
        msg = n + ' env var is not set. Check the .env file.'
        logging.error(msg)
        raise KeyError(msg)

    return d

def get_data_dir() -> str:
    return get_env(DATA_DIR_ENV)

def get_saved_model_dir() -> str:
    return get_env(SAVED_MODEL_DIR_ENV)

def get_val_dataset_path(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, get_env(VAL_DATASET_PATH_ENV))

def get_train_dataset_path(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, get_env(TRAIN_DATASET_PATH_ENV))

def get_images_dir(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, get_env(IMAGES_PATH_ENV))
