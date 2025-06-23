import json, toml, logging, os

from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from harmony_config.productLines import PRODUCTLINES as PLS
from harmony_config.productLines import string_to_productLine
from utils.data_conversion import label_to_json, format_json
from helper.image_processing import get_tensor_from_image

from tensorflow.keras import models 



# given:
#   a productline
# reutrn true if the following are equal (otherwise false):
#   model outputs layer?
#   length of the #_labels.toml
#   config.toml (outputs value)
def validate_outputs(pl: PLS) -> bool:
    # TODO finish this function 
    logging.warning('[validate_outputs] not implemented yet. returning true')
    return True 

# given:
#   an image, 
#   a productline, 
#   and the submodel type (unique identifier for which (sub)model we are using for evaluation)
# return the _id of the image 
def evaluate(image: Image, model_type: int, pl: PLS) -> str:
    if model_type = 0:
        # validate everything to make sure we aren't going to run into an error
        # could remove for efficiency?


        # the output of this evaluation is going to feed into iteself with a recursive call
    else:
        # the output is going to be the real deal (_id)
        # use the master_labels.csv

    # TODO finish this function 
    logging.warning('[evaluate] not implemented yet. returning \'0\'')
    return '0'

# given:
#   a productline
#   and the submodel type (unique identifier for which (sub)model we are using for evaluation)
# return the tensorflow keras model
# TODO: toss this in a utils package if it gets reused later
def get_model(model_type: int, pl: PLS) -> tf.keras.Model:
    # TODO: wrap this in a try_catch?
    model_name = pl.value + '.keras'
    model_path = os.path.join(os.getenv('MODEL_DIR'), model_name)

    return models.load_model(model_path)

