import logging
import time 
import gc

# import matplotlib
# matplotlib.use('TkAgg')  
# import matplotlib.pyplot as plt

import tensorflow as tf 

from tensorflow import keras
from tensorflow.keras import mixed_precision 

from data.collect import download_images_parallel, collect
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 
from data.dataset import generate_datasets
from training.train import train_product_line, continue_train_product_line
from utils.file_handler.toml import *

from data.dataset import * 

logging.getLogger().setLevel(10)

if __name__ == '__main__':
    # clear tensorflow
    keras.backend.clear_session()

    # force garbage collection
    gc.collect()

    # expand all gpus for growth (will slow down the kernel)
    # conservative_gpu_usage()
    
    # makes things faster?
    mixed_precision.set_global_policy("mixed_bfloat16")
    
    # just in time compilation?
    tf.config.optimizer.set_jit(True)


    # ==========================================


    # DOWNLOAD THE IMAGES
    # collect(PLS.LORCANA)

    # GENERATE THE DATASETS
    generate_datasets(PLS.LORCANA)
    
    # TRAIN THE MODEL
    train_product_line(PLS.LORCANA, ['m0'])

    # CONTINUE TRAIN THE MODEL
    # continue_train_product_line(PLS.LORCANA, ['m0'], '2025.08.01_18.25.15')


def conservative_gpu_usage():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        # logical_gpus = tf.config.list_logical_devices('GPU')
        logging.debug(gpus)
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.error(e)

