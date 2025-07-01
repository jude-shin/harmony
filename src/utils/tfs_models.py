import json
import toml
import logging
import os
import typeguard
import requests

import numpy as np

from utils.product_lines import PRODUCTLINES as PLS

def identify(instances: list, model_name: str, pl: PLS) -> list[str]:
    '''
    Identifies a card with multiple models, giving the most confident output.

    Args:
        instances (list): The list of preprocessed images that are converted to a tensor, which is converted to a list
        model_name (string): unique identifier for which (sub)model we are using for evaluation
            ex) in the model "m12.keras", the model_name is "m12"
            ex) in the labels toml "m0_labels.toml", the model_name is "m0"
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        list[str]: a list of the most confident labels of the given images (from the master layer)
    '''

    TFS_PORT = os.getenv('TFS_PORT')
    url = f'http://tfs-{pl.value}:{TFS_PORT}/v1/models/{model_name}:predict'

    response = requests.post(url, json={'instances': instances}).json()

    model_config_dict = get_model_config(pl)
    
    predictions = response['predictions']
    final_prediction_labels = [None] * len(instances)

    if model_config_dict[model_name]['is_final']:
        for i, p in enumerate(predictions):
            best_prediction_label = str(np.argmax(p))
            final_prediction_labels[i] = best_prediction_label 
            logging.info('Model [%s] final prediction for image %d: %s', model_name, i, best_prediction_label)
            return final_prediction_labels

    submodel_inputs = {}
    image_indices_by_submodel = {}

    for i, p in enumerate(predictions):
        best_label = str(np.argmax(p))

        labels_to_model_names_dict = get_model_labels(model_name, pl)
        next_model = labels_to_model_names_dict[best_label]
        logging.info('Model [%s] defers image %d to submodel [%s] (label: %s)', model_name, i, next_model, best_label)
        if next_model not in submodel_inputs:
            submodel_inputs[next_model] = []
            image_indices_by_submodel[next_model] = []

        submodel_inputs[next_model].append(instances[i])
        image_indices_by_submodel[next_model].append(i)

    for next_model, sub_instances in submodel_inputs.items():
        sub_results = identify(sub_instances, next_model, pl)
        for j, result in zip(image_indices_by_submodel[next_model], sub_results):
            final_prediction_labels[j] = result

    return final_prediction_labels 


def get_model_metadata(model_name: str, pl: PLS) -> dict:
    '''
    gets the json dict of the metadata that tensorflow serving returns

    Args:
        model_name (string): unique identifier for which (sub)model we are using for evaluation
            ex) in the model "m12.keras", the model_name is "m12"
            ex) in the labels toml "m0_labels.toml", the model_name is "m0"
        pl (PRODUCTLINES): The product_line we are working with.
    Returns: 
        dict: json dict response from tfs metadata get request
    '''
    # port = get_port(pl) 
    port = os.getenv('TFS_PORT')
    url = f'http://tfs-{pl.value}:{port}/v1/models/{model_name}/metadata'
    return requests.get(url).json()

def get_model_labels(model_name: str, pl: PLS) -> dict:
    '''
    Gets the labels to _id in a hashmap form.
    
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
            logging.error(' [get_model_labels] DATA_DIR env var not set. Returning an empty dict.')
            return {}

        full_toml_path = os.path.join(data_dir, pl.value, toml_path)

        with open(full_toml_path, 'r') as f:
            return toml.load(f)

    except Exception as e:
        logging.error(' [get_model_labels] unexpected error %s. Returning an empty dict.', e)
        return {}

def get_model_config(pl: PLS) -> dict:
    '''
    Gets the tensorflow model config.toml (shows the is_final status and no. of outputs a model has).
    
    Args:
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        dict: the dictonary of the toml (in dictionary form)
    '''
    try:
        toml_path = 'config.toml'
        data_dir = os.getenv('MODEL_DIR')

        if data_dir is None:
            logging.error(' [get_model_config] DATA_DIR env var not set. Returning an empty dict.')
            return {}

        full_toml_path = os.path.join(data_dir, pl.value, toml_path)

        with open(full_toml_path, 'r') as f:
            return toml.load(f)

    except Exception as e:
        logging.error(' [get_model_config] unexpected error %s. Returning an empty dict.', e)
        return {}

# TODO
def validate_config(pl: PLS) -> bool:
    return True 
