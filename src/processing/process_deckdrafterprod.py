import os
import json 
import requests 
import logging
import pickle

from time import time

from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time
from utils.file_handler.json import load_deckdrafterprod 
from utils.file_handler.dir import get_data_dir

# TODO: renme this file to preprocessing or something
# TODO: function that adds images that already have an _id
    # checks to see that the key is already present, (skips the image if the key is unknown)
    # downloads the image with the _id as the name 
    # if there is already an image with that name, add a (n) to the end


# TODO: we are going to add all the photos to the queue side
# all this data is temporary because it will be turned into a tf.data.Dataset
# but maybe we should still save them to reduce the overhead of retraining when the time comes to retrain the whole thing?


def download_images(pl: PLS, size: str):
    '''
    NOTE: this is used only if there are new cards with new _ids added to the deckdrafterprod
    IT IS NOT used for when we want to download images of cards that we know the _id of already

    for each element in a json that is given ()
    check to see if we have the key already (this means that there is already a downloaded image)

    if the key is not registerd, then download an image
    add it to a processed_deckdrafterprod.json
    each name of the card should be the _id
    '''

    images_downloaded: int = 0
    images_skipped: int = 0
    st: float = time()

    deckdrafterprod = load_deckdrafterprod(pl, 'r')

    for card in deckdrafterprod:
        _id: str = card['_id']
        
        logging.info(f'downloading image _id: %s', _id)

        try:
            image_url = card['images'][size]
            data = requests.get(image_url).content

            image_path = os.path.join(images_dir, _id+'.jpg')

            with open(image_path, 'wb+') as f:
                f.write(data)
                images_downloaded += 1

        except KeyError as e:
            #TODO it may be better to do something else.. I don't know what to do about this. maybe take note and have this be in the queue for the next
            # round of retraining
            logging.error(f'[download_images] no images of that size... skippping image for now. Error: ', e)
            images_skipped += 1
            continue
        except ValueError as e:
            logging.error(f'[download_images] some information not found... skipping image for now. Error: ', e)
            images_skipped += 1
            continue


    # TODO: show the number downloaded, and the number of images skipped, and the time it took to download everything 
    logging.info(f'images downloaded: %d', images_downloaded)
    logging.info(f'images skipped: %d', images_skipped)
    logging.info(f'total elapsed time: %s', get_elapsed_time(st))
        

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
    # logging.warning('generate_keys not implemented yet')
    data_dir = get_data_dir()
    deckdrafterprod = load_deckdrafterprod(pl, 'r')

    label_to_id = []
    for card in deckdrafterprod:
        _id = card['_id']
        label_to_id.append(str(_id))

    # TODO: use the file_management module in utils
    pickle_path = 'master_ids.pkl'

    label_to_id_path = os.path.join(data_dir, pl.value, pickle_path)
    with open(label_to_id_path, 'wb+') as f:
        pickle.dump(label_to_id, f)


def process_deckdrafterprod(pl: PLS):
    logging.info('process_deckdrafterprod called...')
    generate_keys(pl)
    download_images(pl, 'large')
