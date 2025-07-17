import logging 

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics

# from training.callbacks import ValidationAccuracyThresholdCallback
from data.dataset import load_record
from cnn.model_structure import CnnModel1, CnnModelClassic15
from utils.file_handler.dir import get_record_path 
from utils.product_lines import PRODUCTLINES as PLS

from training.callbacks import get_callbacks


def train(pl: PLS):
    ###########################
    #   keras_models verson   #
    ###########################
    # make a folder in the keras_models folder
    # this will be based on the time and date
    # training logs, the model.keras, and checkpoiint.keras will be located here

    ############################
    #   Loading the Datasets   #
    ############################
    # load the validation and training datasets from the record stored on disk
    # training data should be multiplied more than the validation data
    # training data should be shuffled and augmented
    # validation can be augmented or shuffled
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=32, shuffle=True, augment=True, multiply=10)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=32, shuffle=False, augment=False, multiply=1)
    logging.info('Finished Loading Validation Dataset!')
   
    #########################
    #   Loading the Model   #
    #########################
    # load the skeleton from cnn/model_structure.py
    # compile the model
    logging.info('Loading Model...')
    model = CnnModelClassic15([437, 313], 994)
    
    # build the layers
    model(tf.zeros([1, 437, 313, 3]))
    
    # compile the model with learning rates and optimizers
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003, beta_1=0.09, beta_2=0.999),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    logging.info('Finished Loading Model!')
    
    ################
    #   Training   #
    ################
    # fit the model with custom callbacks and the datasets we created
    logging.info('Starting training...')
    model.fit(train_ds,
              epochs=10000000000000,
              validation_data=val_ds, 
              callbacks=get_callbacks()
              )

    model.save('NEWMODEL.keras')


def retrain():
    # call train() with small learning for fine tuning
    pass





