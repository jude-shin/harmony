import json, toml, logging, os, typeguard

from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from harmony_config.product_lines import PRODUCTLINES as PLS
from harmony_config.product_lines import string_to_product_line
from utils.data_conversion import label_to_json, format_json
from helper.image_processing import get_tensor_from_image

from tensorflow.keras import models



    # TODO finish this function
def validate_outputs(pl: PLS) -> bool:
    '''
    Validates the configuration files in the model directory, making sure the following are correct:
      model outputs layer?
      length of the #_labels.toml
      config.toml (outputs value)

    Args:
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        bool: Whether all of the checks passed or not
    '''
    logging.warning('[validate_outputs] not implemented yet. returning true')
    return True

def evaluate(image: Image.Image, model_no: int, pl: PLS) -> str:
    '''
    Identifies a card.

    Args:
        image: (Image.Image): The image of the card that is to be identified,
        pl (PRODUCTLINES): The product_line we are working with.
        model_no (int): unique identifier for which (sub)model we are using for evaluation
    Returns: 
        str: _id of the image (based on the deckdrafterprod)
    '''
    if model_no == 0:
        # validate everything to make sure we aren't going to run into an error
        # could remove for efficiency?

        logging.warning('not implemented')

        # the output of this evaluation is going to feed into iteself with a recursive call
    else:
        logging.warning('not implemented')
        # the output is going to be the real deal (_id)
        # use the master_labels.csv

    # TODO finish this function
    logging.warning('[evaluate] not implemented yet. returning \'0\'')
    return '0'

# TODO: toss this in a utils package if it gets reused later
def get_model(model_no: int, pl: PLS) -> models.Model:
    '''
    Gets the tensorflow model.
    Args:
        pl (PRODUCTLINES): The product_line we are working with.
        model_no (int): unique identifier for which (sub)model we are using for evaluation
    Returns:
        models.Model: the trained tensorflow model
    '''
    try:
        model_name: str = 'm' + str(model_no) + '.keras'
        model_dir = os.getenv('MODEL_DIR')
        if model_dir is None:
            logging.error('[get_model] MODEL_DIR env var not set')
            return None
        model_path = os.path.join(model_dir, pl.value, model_name)
        return models.load_model(model_path)
    except Exception as e:
        logging.error('[get_model] unexpected error %s', e)
        return None
