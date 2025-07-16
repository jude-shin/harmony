import logging
import time 
import gc

import tensorflow as tf 
from tensorflow import keras

from data.collect import download_images_parallel, collect
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 

from data.dataset import generate_datasets

from training.train import train 

logging.getLogger().setLevel(20)

if __name__ == '__main__':
    # clear tensorflow
    keras.backend.clear_session()

    # force garbage collection
    gc.collect()

    # st = time.time()

    # collect(PLS.POKEMON)
    # generate_datasets(PLS.POKEMON)

    # logging.warning(' ----> ELAPSED TIME: ' + get_elapsed_time(st))
    
    train(PLS.LORCANA)
