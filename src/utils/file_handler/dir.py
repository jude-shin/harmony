import os
import logging 

DATA_DIR = 'DATA_DIR'
SAVED_MODEL_DIR = 'SAVED_MODEL_DIR'
TRAIN_DATASET_DIR = 'TRAIN_DATASET_DIR'
VAL_DATASET_DIR = 'VAL_DATASET_DIR'

def get_dir(n: str) -> str:
    d = os.getenv(n)
    if d is None:
        msg = n + ' env var is not set. Check the .env file.'
        logging.error(msg)
        raise KeyError(msg)

    return d

def get_data_dir() -> str:
    return get_dir(DATA_DIR)

def get_saved_model_dir() -> str:
    return get_dir(SAVED_MODEL_DIR)

def get_val_dataset_dir() -> str:
    return get_dir(TRAIN_DATASET_DIR)

def get_train_dataset_dir() -> str:
    return get_dir(VAL_DATASET_DIR)


