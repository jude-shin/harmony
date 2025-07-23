import os
import logging 
import time

from utils.product_lines import PRODUCTLINES as PLS

DATA_DIR_ENV= 'DATA_DIR'
SAVED_MODEL_DIR_ENV = 'SAVED_MODEL_DIR'
KERAS_MODEL_DIR_ENV = 'KERAS_MODEL_DIR'
CONFIG_DIR_ENV = 'CONFIG_DIR'

# TRAIN_DATASET_PATH_ENV = 'TRAIN_DATASET_PATH_NAME'
# VAL_DATASET_PATH_ENV = 'VAL_DATASET_PATH_NAME'
RECORD_PATH_ENV = 'RECORD_PATH_NAME'
IMAGES_DIR_ENV = 'IMAGES_DIR_NAME'
CONFIG_PATH_ENV = 'CONFIG_PATH_NAME'

def get_env(n: str) -> str:
    d = os.getenv(n)
    if d is None:
        msg = n + ' env var is not set. Check the .env file.'
        logging.error(msg)
        raise KeyError(msg)

    return d
# -------------------------
def get_data_dir() -> str:
    return get_env(DATA_DIR_ENV)

def get_saved_model_dir() -> str:
    return get_env(SAVED_MODEL_DIR_ENV)

def get_keras_model_dir() -> str:
    return get_env(KERAS_MODEL_DIR_ENV)

def get_config_dir() -> str:
    return get_env(CONFIG_DIR_ENV)

# -------------------------

def get_record_path(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, get_env(RECORD_PATH_ENV))

def get_images_dir(pl: PLS) -> str:
    return os.path.join(get_data_dir(), pl.value, get_env(IMAGES_DIR_ENV))

def get_config_path(pl: PLS) -> str:
    return os.path.join(get_config_dir(), pl.value, get_env(CONFIG_PATH_ENV))
