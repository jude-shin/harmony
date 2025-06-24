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

def identify(image: Image.Image, model_name: str, pl: PLS) -> str:
    '''
    Identifies a card with multiple models, giving the most confident output

    Args:
        image: (Image.Image): The image of the card that is to be identified,
        label_no (int): TODO FINISH THIS DEF
        pl (PRODUCTLINES): The product_line we are working with.
    Returns: 
        str: the most confident label of the image (from the master layer)
    '''
    
    # model_name = 'm' + str(label_no)

    model = get_model(model_name, pl)
    best_prediction_label = evaluate(image, model)
    print('\nBEST MODEL PREDICTION: %s' % best_prediction_label)

    # TODO : check the config.toml to see if the model name is a "master model" (if it is, then we can toss it in here)
    if model_name == 'm0':
        # the best prediction should be the output of the model
        labels_to_model_names = get_model_config(model_name, pl)
        next_label_name = labels_to_model_names[best_prediction_label]

        # the output of this evaluation is going to feed into iteself with a recursive call
        return identify(image, next_label_name, pl)

    # the output is going to be the real deal (_id)
    return best_prediction_label

def evaluate(image: Image.Image, model: models.Model) -> str:
    '''
    Feed the model the inputs, and get the most confident direct output (no interpretation of the output) 

    Args:
        image: (Image.Image): The image of the card that is to be identified,
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

    logging.info('confidence: %s', confidence)
    logging.info('best_prediction: %s', best_prediction_label)

    return str(best_prediction_label)



# TODO: toss this in a utils package if it gets reused later
def get_model(model_name: str, pl: PLS) -> models.Model:
    '''
    Gets the tensorflow model.
    Args:
        pl (PRODUCTLINES): The product_line we are working with.
        model_name (string): unique identifier for which (sub)model we are using for evaluation
            ex) in the model "m12.keras", the model_name is "m12"
            ex) in the labels toml "m0_labels.toml", the model_name is "m0"
    Returns:
        models.Model: the trained tensorflow model
    '''
    try:
        model_path = model_name + '.keras'
        model_dir = os.getenv('MODEL_DIR')

        if model_dir is None:
            logging.error('[get_model] MODEL_DIR env var not set')
            return None
        full_model_path = os.path.join(model_dir, pl.value, model_path)
        return models.load_model(full_model_path)
    except Exception as e:
        logging.error('[get_model] unexpected error %s', e)
        return None


# TODO: toss this in a utils package if it gets reused later
def get_model_config(model_name: str, pl: PLS) -> dict:
    '''
    Gets the tensorflow model.
    Args:
        pl (PRODUCTLINES): The product_line we are working with.
        model_name (string): unique identifier for which (sub)model we are using for evaluation
            ex) in the model "m12.keras", the model_name is "m12"
            ex) in the labels toml "m0_labels.toml", the model_name is "m0"
    Returns:
        dict: the dictonary of labels to model_names 
    '''
    try:
        toml_path = model_name + '_labels.toml'
        data_dir = os.getenv('DATA_DIR')

        if data_dir is None:
            logging.error('[get_model_config] DATA_DIR env var not set')
            return None

        full_toml_path = os.path.join(data_dir, pl.value, toml_path)

        with open(full_toml_path, 'r') as f:
            return toml.load(f)

    except Exception as e:
        logging.error('[get_model] unexpected error %s', e)
        return None

