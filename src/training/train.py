import logging 

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics

# from training.callbacks import ValidationAccuracyThresholdCallback
from data.dataset import load_record
from cnn.model_structure import CnnModel1
from utils.file_handler.dir import get_record_path 
from utils.product_lines import PRODUCTLINES as PLS

from training.callbacks import get_callbacks


def train(pl: PLS):
    # generate a dataset from the queue

    # (1) load the validation and training datasets from the record
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=64, shuffle=True, augment=True, multiply=10)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=64, shuffle=False, augment=False, multiply=2)
    logging.info('Finished Loading Validation Dataset!')

    # (2) load the model
    logging.info('Loading Model...')

    # create object 
    model = CnnModel1([437, 313], 994) # (NOT 993 because one of them were skipped, but we still want that entry...)
    
    # build the layers
    model(tf.zeros([1, 437, 313, 3]))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003, beta_1=0.09, beta_2=0.999),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    logging.info('Finished Loading Model!')

    # (3) model.fit with the custom callbacks
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





