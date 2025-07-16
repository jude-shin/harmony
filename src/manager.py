import logging
import time 

from data.collect import download_images_parallel, collect
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 

from data.dataset import generate_datasets

from training.train import train 

logging.getLogger().setLevel(20)

if __name__ == '__main__':
    # st = time.time()

    # collect(PLS.POKEMON)
    # generate_datasets(PLS.POKEMON)

    # logging.warning(' ----> ELAPSED TIME: ' + get_elapsed_time(st))
    
    train()
