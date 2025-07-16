import logging 

import tensorflow as tf

from tensorflow import keras

# from training.callbacks import ValidationAccuracyThresholdCallback
from data.dataset import load_record
from cnn.model_structure import CnnModel1
from utils.file_handler.dir import get_record_path
from utils.product_lines import PRODUCTLINES as PLS


def train(pl: PLS):
    # generate a dataset from the queue

    # (1) load the validation and training datasets from the record
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=32, shuffle=True, augment=True, multiply=10)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=32, shuffle=False, augment=False, multiply=1)
    logging.info('Finished Loading Validation Dataset!')

    # (2) load the model
    logging.info('Loading Model...')

    # create object 
    model = CnnModel1([312, 413], 994) # (NOT 993 because one of them were skipped, but we still want that entry...)

    # # normalize layers
    # model.preprocess.normalize_layer.adapt(train_ds.map(lambda x, y: x))

    # build the layers
    model(tf.zeros([1, 312, 413, 3]))


    # model.compile(
    #         optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    #         loss=keras.losses.BinaryCrossentropy(),
    #         metrics=[
    #             keras.metrics.BinaryAccuracy(),
    #             keras.metrics.FalseNegatives(),
    #             ],
    #         )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )



    logging.info('Finished Loading Model!')

    # (3) model.fit with the custom callbacks
    logging.info('Starting training...')
    model.fit(train_ds,
              epochs=25,
              validation_data=val_ds, 
              )


def retrain():

    # call train() with small learning for fine tuning
    pass





