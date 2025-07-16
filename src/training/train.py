import logging 

import tensorflow as tf

from tensorflow import keras

from training.callbacks import ValidationAccuracyThresholdCallback
from data.dataset import load_record
from cnn.model_structure import CnnModel1


def train():
    # generate a dataset from the queue

    # usage

    # (1) load the model
    logging.info('Loading Model...')
    model = CnnModel1([312, 413], 993)
    model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.FalseNegatives(),
                ],
            )
    logging.info('Finished Loading Model!')


    # (2) load the validation and training datasets from the record
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record('record.tfrecord', batch_size=32, shuffle=True, augment=True, multiply=10)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record('record.tfrecord', batch_size=32, shuffle=False, augment=False, multiply=1)
    logging.info('Finished Loading Validation Dataset!')

    # (3) model.fit with the custom callbacks
    logging.info('Starting training...')
    model.fit(train_ds,
              epochs=25,
              validation_data=val_ds, 
              )


def retrain():

    # call train() with small learning for fine tuning
    pass





