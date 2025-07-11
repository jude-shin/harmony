import logging
import time 

from data.collect import download_images_parallel
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 

from processing.generate_datasets import generate_datasets

logging.getLogger().setLevel(20)

if __name__ == '__main__':
    st = time.time()

    download_images_parallel(pl=PLS.POKEMON, max_workers=64)
    
    logging.warning(' ----> ELAPSED TIME: ' + get_elapsed_time(st))
    # generate_datasets(pl=PLS.LORCANA)
