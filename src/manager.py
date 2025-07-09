import logging

from utils.images import download_images_parallel
from utils.product_lines import PRODUCTLINES as PLS

logging.getLogger().setLevel(20)

if __name__ == '__main__':
    download_images_parallel(pl=PLS.POKEMON, max_workers=16)
