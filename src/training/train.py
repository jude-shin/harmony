import logging 
import os

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics, layers, Sequential
from time import localtime, strftime

# from training.callbacks import ValidationAccuracyThresholdCallback
from data.dataset import load_record
from cnn.model_structure import CnnModel1, CnnModelClassic15
from utils.file_handler.dir import get_record_path, get_keras_model_dir
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_current_time

from training.callbacks import get_callbacks


from cnn.sequential_models import model_classic_1, model_classic_15


def train(pl: PLS):
    ###########################
    #   keras_models verson   #
    ###########################
    # make a folder in the keras_models folder
    # this will be based on the time and date
    # training logs, the model.keras, and checkpoiint.keras will be located here

    # make the dir name based on the time
    keras_model_dir = strftime('%Y.%m.%d_%H.%M.%S', localtime())
    keras_model_dir = os.path.join(get_keras_model_dir(), pl.value, keras_model_dir)

    # if the directory exsists, log it as a warning, because I believe it will be overwritten
    if os.path.exists(keras_model_dir):
        logging.warning('keras model path exists already... data may be overwritten')

    os.mkdir(keras_model_dir)



    ############################
    #   Loading the Datasets   #
    ############################
    # load the validation and training datasets from the record stored on disk
    # training data should be multiplied more than the validation data
    # training data should be shuffled and augmented
    # validation can be augmented or shuffled
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=64, shuffle=True, augment=True, multiply=50)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=64, shuffle=False, augment=False, multiply=1)
    logging.info('Finished Loading Validation Dataset!')
   

    #########################
    #   Loading the Model   #
    #########################
    # load the skeleton from cnn/model_structure.py
    # compile the model
    logging.info('Loading Model...')
    keras_model = CnnModelClassic15([437, 313], 994)
    # keras_model = model_classic_15(437, 313, 994)
    
    # build the layers
    keras_model(tf.zeros([1, 437, 313, 3]))
    
    # compile the model with learning rates and optimizers
    keras_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
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

    keras_model_path = os.path.join(keras_model_dir, 'model.keras')
    keras_model.save(keras_model_path)


def retrain():
    # call train() with small learning for fine tuning
    pass

