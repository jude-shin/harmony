import json
import toml
import logging
import os
import typeguard

from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from harmony_config.product_lines import PRODUCTLINES as PLS
from harmony_config.product_lines import string_to_product_line
from utils.data_conversion import label_to_json, format_json
from helper.image_processing import get_tensor_from_image

from tensorflow.keras import models

# # TODO finish this function
# def validate_outputs(pl: PLS) -> bool:
#     '''
#     Validates the configuration files in the model directory, making sure the following are correct:
#       model outputs layer?
#       length of the #_labels.toml
#       config.toml (outputs value)
# 
#     Args:
#         pl (PRODUCTLINES): The product_line we are working with.
#     Returns:
#     '''
# 
#     # TODO : this code will be the one that is throwing assertion errors
#     logging.warning('[validate_outputs] not implemented yet. checking nothing...')

def identify(image: Image.Image, model_no: int, pl: PLS) -> str:
    '''
    Identifies a card with multiple models, giving the most confident output

    Args:
        image: (Image.Image): The image of the card that is to be identified,
        model_no (int): unique identifier for which (sub)model we are using for evaluation
        pl (PRODUCTLINES): The product_line we are working with.
    Returns: 
        str: the most confident label of the image (from the master layer)
    '''

    logging.info('Model Number: %d', model_no)

    model = get_model(model_no, pl)

    best_prediction_label = evaluate(image, model_no, model)

    if model_no == 0:
        # the best prediction should be the output of the model
        next_model_no = int(best_prediction_label)

        # the output of this evaluation is going to feed into iteself with a recursive call
        return identify(image, next_model_no, pl)

    # the output is going to be the real deal (_id)
    return best_prediction_label

def evaluate(image: Image.Image, model_no: int, model: models.Model) -> str:
    '''
    Feed the model the inputs, and get the most confident direct output (no interpretation of the output) 

    Args:
        image: (Image.Image): The image of the card that is to be identified,
        model_no (int): unique identifier for which (sub)model we are using for evaluation
        pl (PRODUCTLINES): The product_line we are working with.
        model (models.Model): the model that is going to do the work
    Returns: 
        # str: the most confident output from the given model 
    '''
    _, model_img_width, model_img_height, _ = model.input_shape

    img_tensor = get_tensor_from_image(image, model_img_width, model_img_height)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    prediction_labels = model.predict(img_tensor)
    best_prediction_label, confidence = np.argmax(
        prediction_labels), prediction_labels[0, np.argmax(prediction_labels)]

    logging.info('model no: %s', model_no)
    logging.info('confidence: %s', confidence)
    logging.info('best_prediction: %s', best_prediction_label)

    return str(best_prediction_label)



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
