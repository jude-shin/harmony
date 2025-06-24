import json, logging, os, typeguard

import pandas as pd
import numpy as np 

from pathlib import Path

from harmony_config.product_lines import PRODUCTLINES as PLS

DATA_DIR = os.getenv('DATA_DIR')

# label: what the tensorflow model will spit out
# _id: deckdrafterprod unique _id
#   allows us to get more information about the card


def label_to_id(label : int, g : PLS) -> str:
    '''
    Convert a given label to the deckdrafterprod _id based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: _id that is associated with that label
    '''
    json = label_to_json(label, g)

    # extract only the _id from the json object
    return ''

def id_to_label(_id : str, g : PLS) -> str:
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
def label_to_json(label : int, g : PLS) -> str:
    '''
    Look up which json object has the particular label based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: json entry that is associated with that label
    '''

    # master-labels.csv (for label -> _id)

    if DATA_DIR is None:
        logging.error('[label_to_json] DATA_DIR env var not set')
        return ''
    master_labels_path = Path(DATA_DIR, g.value, 'master_labels.csv')
    master_labels = pd.read_csv(master_labels_path)

    predicted_id = master_labels[master_labels['label'] == label]['_id'].iloc[0]

    # deckdrafterprod.json (for _id -> various information)
    deckdrafterprod_path = Path(DATA_DIR, g.value, 'deckdrafterprod.json')
    with open(deckdrafterprod_path, 'r') as deckdrafterprod_file:
        deckdrafterprod = json.load(deckdrafterprod_file)

    card_obj = {}
    for obj in deckdrafterprod:
        if str(obj['_id']) == str(predicted_id):
            logging.info('object with {_id} FOUND!')
            card_obj = obj
    if card_obj == {}:
        logging.warning('[label_to_json] object with %s not found... returning empty json object', str(predicted_id))

    # returns the raw object without any filtering of the fields 
    # each field name can be different based on the game, so we must process it
    return json.dumps(card_obj)


# TODO: make json_string a dict type? (we don't need to keep converting it and stuff)
# TODO: make return type a dict as well?
def format_json(json_string : str, g : PLS) -> str:
    '''
    Formats the raw json object (see label_to_json) with infomation the api is looking for 

    Args:
        json_string (str): Json object string that is 
        pl (PRODUCTLINES): The producteLine we are working with.
    Returns:
        str: formatted json object
    '''
    try:
        formatted_json = {}
        data = json.loads(json_string)
        if g == PLS.LORCANA:
            formatted_json = {
                'name': data['productName'],
                '_id': data['_id'],
                'image_url': data['images']['large'],
                'tcgplayer_id': data['tcgplayer_productId'], 
                # TODO: this gives the first price that it finds in the listings
                # maybe take an average? or get the lowest one?
                'price': data['listings'][0]['price'],
                }
        # elif g == PRODUCTLINES.MTG:
        # elif g == PRODUCTLINES.POKEMON:
        else: raise ValueError()
        return json.dumps(formatted_json)
    except KeyError:
        logging.warning('[format_json] key not found... returning empty json object')
        return json.dumps({})
    except ValueError:
        logging.warning('Game Type %s not supported', g.value)
        return json.dumps({})
