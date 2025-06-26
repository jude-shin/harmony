import json
import logging
import os
import typeguard
import toml

import pandas as pd
import numpy as np 

from pathlib import Path

from data_defs.product_lines import PRODUCTLINES as PLS

DATA_DIR = os.getenv('DATA_DIR')

# label: what the tensorflow model will spit out
# _id: deckdrafterprod unique _id
#   allows us to get more information about the card


def label_to_id(label : int, pl : PLS) -> str:
    '''
    Convert a given label to the deckdrafterprod _id based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: _id that is associated with that label
    '''
    json = label_to_json(label, pl)

    # extract only the _id from the json str 
    return ''

def id_to_label(_id : str, pl : PLS) -> str:
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

# TODO: make return type a dict as well?
def label_to_json(label : int, pl : PLS) -> str:
    '''
    Look up which json str has the particular label based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: json entry that is associated with that label
    '''

    # master-labels.csv (for label -> _id)
    if DATA_DIR is None:
        logging.error(' [label_to_json] DATA_DIR env var not set. Returning empty str.')
        return ''
    master_labels_path = os.path.join(DATA_DIR, pl.value, 'master_labels.toml')
    with open(master_labels_path, 'r') as f:
        master_labels = toml.load(f)

    predicted_id = master_labels[str(label)]

    logging.info(' Label: %d -> _id: %s', label, predicted_id)

    # deckdrafterprod.json (for _id -> various information)
    deckdrafterprod_path = Path(DATA_DIR, pl.value, 'deckdrafterprod.json')
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
    return json.dumps(card_obj)


# TODO: make json_string a dict type? (we don't need to keep converting it and stuff)
# TODO: make return type a dict as well?
def format_json(json_string : str, pl : PLS) -> str:
    '''
    Formats the raw json object (see label_to_json) with infomation the api is looking for 

    Args:
        json_string (str): Json str that is 
        pl (PRODUCTLINES): The producteLine we are working with.
    Returns:
        str: formatted json str 
    '''
    # TODO : I think this should go into the data definitions section...
    # TODO : we also don't need all this information. we only need the _id, we may get to trash this entire function
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
