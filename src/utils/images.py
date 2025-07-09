import os
import logging
import time 
import requests

from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import Timeout, RequestException

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.json import load_deckdrafterprod

def download_image(item, index, images_dir, max_retries=5, backoff_base=2):
    try:
        _id = item['_id']
        url = item['images']['large']
    except KeyError as e:
        logging.warning(f'[{index}] Missing _id or url. Skipping. Item: {item}')
        return

    filename = os.path.join(images_dir, f'{_id}.jpg')

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=(5, 10))
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            logging.info(f'[{index}] Downloaded: {_id}')
            return
        except Timeout as e:
            logging.warning(f'[{index}] Timeout (attempt {attempt + 1}) for {_id}: {e}')
        except RequestException as e:
            logging.warning(f'[{index}] Request error (attempt {attempt + 1}) for {_id}: {e}')
        except Exception as e:
            logging.error(f'[{index}] Unexpected error (attempt {attempt + 1}) for {_id}: {e}')

        if attempt < max_retries - 1:
            delay = backoff_base ** attempt
            time.sleep(delay)

    logging.error(f'[{index}] Failed to download {_id} after {max_retries} attempts')


def download_images_parallel(pl: PLS, max_workers=8):
    deckdrafterprod = load_deckdrafterprod(pl, 'r')

    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        msg = 'DATA_DIR env var is not set...'
        logging.error(msg)
        raise KeyError(msg)

    images_dir = os.path.join(data_dir, pl.value, 'images')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        logging.info(f'Created output directory: {images_dir}')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, item in enumerate(deckdrafterprod):
            executor.submit(download_image, item, idx, images_dir)
        executor.shutdown(wait=True)
