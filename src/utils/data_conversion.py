import json
import logging
import os
import typeguard
import toml

import pandas as pd
import numpy as np 

from pathlib import Path

from utils.product_lines import PRODUCTLINES as PLS

def label_to_id(label : int, pl : PLS) -> str:
    '''
    Convert a given label to the deckdrafterprod _id based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: _id that is associated with that label
    '''
    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        logging.error(' [label_to_json] DATA_DIR env var not set. Returning empty str.')
        return ''
    master_labels_path = os.path.join(data_dir, pl.value, 'master_labels.toml')
    with open(master_labels_path, 'r') as f:
        master_labels = toml.load(f)

    return master_labels[str(label)]


def id_to_label(_id : str, pl : PLS) -> str:
    # TODO
    '''
    Convert a given deckdrafterprod _id to the label based on the master_labels.toml

    Args:
        _id (str): deckdrafterprod _id 
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: what the tensorflow model will spit out
    '''
    
    # look up the variable
    return ''

def label_to_json(label : int, pl : PLS) -> dict:
    '''
    Look up which json str has the particular label based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        dict: json entry that is associated with that label (dict by default)
    '''

    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        logging.error(' [label_to_json] DATA_DIR env var not set. Returning empty str.')
        return ''
    master_labels_path = os.path.join(data_dir, pl.value, 'master_labels.toml')
    with open(master_labels_path, 'r') as f:
        master_labels = toml.load(f)

    predicted_id = master_labels[str(label)]

    logging.info(' Label: %d -> _id: %s', label, predicted_id)

    # deckdrafterprod.json (for _id -> various information)
    deckdrafterprod_path = Path(data_dir, pl.value, 'deckdrafterprod.json')
    with open(deckdrafterprod_path, 'r') as deckdrafterprod_file:
        deckdrafterprod = json.load(deckdrafterprod_file)

    card_obj = {}
    for obj in deckdrafterprod:
        if str(obj['_id']) == str(predicted_id):
            logging.info(' Json object with _id: %s found in deckdrafterprod', predicted_id)
            card_obj = obj
    if card_obj == {}:
        logging.warning(' [label_to_json] object with %s not found. Returning empty json str', predicted_id)

    # returns the raw json str without any filtering of the fields 
    # each field name can be different based on the game, so we must process it
    return card_obj


# TODO: make json_string a dict type? (we don't need to keep converting it and stuff)
# TODO: make return type a dict as well?
def format_json(json_string : str, pl : PLS) -> str:
    '''
    Formats the raw json object (see label_to_json) with infomation the api is looking for 
    DEPRECIATED (basically useless to me but ill keep it here if we want it)

    Args:
        json_string (str): Json str that is 
        pl (PRODUCTLINES): The producteLine we are working with.
    Returns:
        str: formatted json str 
    '''

    try:
        formatted_json = {}
        data = json.loads(json_string)
        if pl == PLS.LORCANA:
            formatted_json = {
                'name': data['productName'],
                '_id': data['_id'],
                # 'image_url': data['images']['large'],
                'tcgplayer_id': data['tcgplayer_productId'], 
                # maybe take an average? or get the lowest one?
                # 'price': data['listings'][0]['price'],
                # 'price': data['medianPrice'],
                }
        elif pl == PLS.POKEMON:
            formatted_json = {
                'name': data['name'],
                '_id': data['_id'],
                # 'image_url': data['images']['large'],
                'tcgplayer_id': data['tcgplayer_productId'], 
                # maybe take an average? or get the lowest one?
                # 'price': data['listings'][0]['price'],
                # 'price': data['medianPrice'],
                }

        # elif pl == PLS.MTG:
        else: raise ValueError()
        return json.dumps(formatted_json)
    except KeyError as e:
        logging.warning(' [format_json] key not found. Returning empty json str. Error: %s', e)
        return json.dumps({})
    except ValueError as e:
        logging.warning(' [format_json] PRODUCTLINE %s not supported. Returning empty json str. Error: %s', pl.value, e)
        return json.dumps({})
