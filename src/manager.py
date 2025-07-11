import logging

from utils.images import download_images_parallel
from utils.product_lines import PRODUCTLINES as PLS

from processing.generate_datasets import generate_datasets

logging.getLogger().setLevel(20)

if __name__ == '__main__':
    download_images_parallel(pl=PLS.POKEMON, max_workers=32)
    # generate_datasets(pl=PLS.LORCANA)
