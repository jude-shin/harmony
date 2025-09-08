import os 
import logging
import time 
import requests
import pickle

from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import Timeout, RequestException

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.json import load_deckdrafterprod
from utils.file_handler.dir import get_data_dir
from utils.file_handler.toml import * 

def collect(pl: PLS):
    generate_keys(pl)
    download_images_parallel(pl, 64)


#############################################
#   images (downloads images in parallel)   #
#############################################
def download_image(item, i, img_keys, images_dir, max_retries=5, backoff_base=2):
    try:
        _id = item['_id']
        # TODO: have a list of keys for the config that match to the product lines (yugioh uses 'card_images' for some horrible reason)
        # url = item['card_images']['image_url']
        url = item
        for k in img_keys:
          # will simulate item['card_images']['image_url']
          url = url.get(k)

        if url is None: raise KeyError

    except KeyError as e:
        logging.warning(f'[{i}] Missing _id or url. Skipping. Item: {item}')
        return

    filename = os.path.join(images_dir, f'{_id}.jpg')

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=(5, 10))
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            logging.info(f'[{i}] Downloaded: {_id}')
            return
        except Timeout as e:
            logging.warning(f'[{i}] Timeout (attempt {attempt + 1}) for {_id}: {e}')
        except RequestException as e:
            logging.warning(f'[{i}] Request error (attempt {attempt + 1}) for {_id}: {e}')
        except Exception as e:
            logging.error(f'[{i}] Unexpected error (attempt {attempt + 1}) for {_id}: {e}')

        if attempt < max_retries - 1:
            delay = backoff_base ** attempt
            time.sleep(delay)

    logging.error(f'[{i}] Failed to download {_id} after {max_retries} attempts')


def download_images_parallel(pl: PLS, max_workers):
    deckdrafterprod = load_deckdrafterprod(pl, 'r')
    
    config = load_model_config(pl)
    config = config['m0']
    img_keys = config['deckdrafterprod_img_keys']

    data_dir = get_data_dir()
    images_dir = os.path.join(data_dir, pl.value, 'images')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        logging.info(f'Created output directory: {images_dir}')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(deckdrafterprod):
            executor.submit(download_image, item, i, img_keys , images_dir)
        executor.shutdown(wait=True)


#############################################
#   keys (processes deckdrafterprod.json)   #
#############################################

def generate_keys(pl: PLS):
    '''
    label_to_id 
        NOTE: if we need the id_to_label, you can load the json to a dict, and then zip the values and the keys
        inverted_dict = dict(zip(original_dict.values(), original_dict.keys()))
    this will be the master key that can be referenced across all models and all versions
    if we loose that file we are kind of screwed
    we should definitly make backups of this

    format can be json, or anything that can be parsed to a hashmap
    '''
    data_dir = get_data_dir()
    deckdrafterprod = load_deckdrafterprod(pl, 'r')

    label_to_id = []
    for card in deckdrafterprod:
        _id = card['_id']
        label_to_id.append(str(_id))


    # TODO: use the file_management module in utils
    # TODO: add the model name in the generate_keys as a parameter
    # pickle_path = '{prefix}_ids.pkl'
    pickle_path = 'm0_ids.pkl'

    label_to_id_path = os.path.join(data_dir, pl.value, pickle_path)
    with open(label_to_id_path, 'wb+') as f:
        pickle.dump(label_to_id, f)
        
