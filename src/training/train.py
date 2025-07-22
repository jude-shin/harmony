import logging 
import os

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics
from time import localtime, strftime

from data.dataset import load_record
from cnn.model_structure import * 
from utils.file_handler.dir import get_record_path, get_keras_model_dir
from utils.file_handler.toml import load_config, save_config
from utils.product_lines import PRODUCTLINES as PLS
from training.callbacks import get_callbacks

# from cnn.sequential_models import model_classic_1, model_classic_15
def train_product_line(pl: PLS, models: list[str]):
    # load the config.toml based on the model and product line
    config = load_config(pl)

    if not models: # if the list is empty, default and train all of the models in the config
        models = list(config.keys())

    # TODO: make these run in parallel?
    for model in models:
        model_config = config[model]
        train_model(pl, model, model_config)

    



def train_model(pl: PLS, model: str, config: dict):
    #################
    #   Variables   #
    #################

    # Training Augmentation Multiplication
    batch_size = config['batch_size']
    model_name = config['model_name']
    img_height = config['img_height']
    img_width = config['img_width']
    num_classes = config['num_unique_classes']
    augment_multiplication = config['augment_multiplication']
    learning_rate = config['learning_rate']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    label_smoothing = config['label_smoothing']

    # TODO
    stopping_threshold = config['stopping_threshold']

    # Callbacks?
    #   Stopping threshold (at 98 or so for val accuracy)


    ###########################
    #   keras_models verson   #
    ###########################

    # make a folder in the keras_models folder
    # this will be based on the time and date
    # training logs, the model.keras, and checkpoiint.keras will be located here

    # make the dir name based on the time
    keras_model_dir = strftime('%Y.%m.%d_%H.%M.%S', localtime())
    keras_model_dir = os.path.join(get_keras_model_dir(), pl.value, keras_model_dir, model)

    # if the directory exsists, log it as a warning, because I believe it will be overwritten
    if os.path.exists(keras_model_dir):
        logging.warning('keras model path exists already... data may be overwritten')

    os.makedirs(keras_model_dir)
    
    #############################
    #   Copy the Model Config   #
    #############################
    
    save_config(keras_model_dir, model, config)

    ############################
    #   Loading the Datasets   #
    ############################

    # load the validation and training datasets from the record stored on disk
    # training data should be multiplied more than the validation data
    # training data should be shuffled and augmented
    # validation can be augmented or shuffled
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=True, augment=True, multiply=augment_multiplication, num_classes=num_classes)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=False, augment=True, multiply=1, num_classes=num_classes)
    logging.info('Finished Loading Validation Dataset!')
   

    #########################
    #   Loading the Model   #
    #########################

    # compile the model
    logging.info('Loading Model...')
    keras_model = parse_model_name(model_name, [img_height, img_width], num_classes)
    
    # build the layers
    keras_model(tf.zeros([1, img_height, img_width, 3]))
    
    # compile the model with learning rates and optimizers
    keras_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
        loss=losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing), # Label smoothing
        metrics=[metrics.CategoricalAccuracy()]
    )

    logging.info('Finished Loading Model!')
   

    ################
    #   Training   #
    ################

    # fit the model with custom callbacks and the datasets we created
    logging.info('Starting training...')
    keras_model.fit(train_ds,
              epochs=10000000000000,
              validation_data=val_ds, 
              callbacks=get_callbacks(keras_model_dir)
              )

    keras_model_path = os.path.join(keras_model_dir, model+'.keras')
    keras_model.save(keras_model_path)


def retrain():
    # call train() with small learning for fine tuning
    pass

