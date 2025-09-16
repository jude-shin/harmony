import logging 
import os
import tomllib

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics, models
from time import localtime, strftime

from data.dataset import load_record
from cnn.model_structure import * 
from utils.file_handler.dir import get_record_path, get_keras_model_dir
from utils.file_handler.toml import * 
from utils.product_lines import PRODUCTLINES as PLS
from training.callbacks import get_callbacks
from utils.version import generate_unique_version

def train_product_line(pl: PLS, models: list[str]):
    # load the config.toml based on the model and product line
    config = load_model_config(pl)

    if not models: # if the list is empty, default and train all of the models in the config
        models = list(config.keys())

    # TODO: make these run in parallel?
    for model in models:
        model_config = config[model]
        train_model(pl, model, model_config)

def continue_train_product_line(pl: PLS, models: list[str], version: str):
    for model in models:
        # load the config that is stored in the appropriate directory
        continue_train_model(pl, model, version)

def train_model(pl: PLS, model: str, config: dict):
    ###########################
    #   keras_models verson   #
    ###########################
    # make the dir name based on the time
    keras_model_dir = strftime('%Y.%m.%d_%H.%M.%S', localtime())
    keras_model_dir = os.path.join(get_keras_model_dir(), pl.value, keras_model_dir, model)

    # if the directory exsists, log it as a warning, because I believe it will be overwritten
    if os.path.exists(keras_model_dir):
        logging.warning('keras model path exists already... data may be overwritten')

    os.makedirs(keras_model_dir)


    #################
    #   Variables   #
    #################

    # Training Augmentation Multiplication
    batch_size = config['batch_size']
    model_name = config['model_name']
    img_height = config['img_height']
    img_width = config['img_width']
    num_classes = config['num_unique_classes']
    multiply = config['augment_multiplication']
    learning_rate = config['learning_rate']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    label_smoothing = config['label_smoothing']

    stopping_threshold = config['stopping_threshold']
   

    #############################
    #   Copy the Model Config   #
    #############################
    
    save_model_config(keras_model_dir, model, config)

    # distribute the workload across ALL gpus
    strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
            )

    with strategy.scope():
        ############################
        #   Loading the Datasets   #
        ############################
        # load the validation and training datasets from the record stored on disk
        # training data should be multiplied more than the validation data
        # training data should be shuffled and augmented
        # validation can be augmented or shuffled

        # =====================================================
        logging.info('Loading Training Dataset from TFRecord...')
        train_ds = load_record(pl, batch_size=batch_size, shuffle=True, multiply=multiply, num_classes=num_classes, img_height=img_height, img_width=img_width, model='m0')
        logging.info('Finished Loading Training Dataset!')

        logging.info('Loading Validation Dataset from TFRecord...')
        val_ds = load_record(pl, batch_size=batch_size, shuffle=True, multiply=1, num_classes=num_classes, img_height=img_height, img_width=img_width, model='m0')
        logging.info('Finished Loading Validation Dataset!')
        # =====================================================
        # logging.warning("⚠ Using synthetic data (no disk I/O).")
        # def make_fake_batch():
        #     images = tf.random.uniform(
        #         shape=[batch_size, img_height, img_width, 3],
        #         minval=0, maxval=1, dtype=tf.float32
        #     )
        #     labels = tf.random.uniform(
        #         shape=[batch_size],
        #         minval=0, maxval=num_classes, dtype=tf.int32
        #     )
        #     labels = tf.one_hot(labels, num_classes)
        #     return images, labels

        # train_ds = (tf.data.Dataset
        #             .from_tensors(make_fake_batch())
        #             .repeat()
        #             .prefetch(tf.data.AUTOTUNE))

        # val_ds = (tf.data.Dataset
        #             .from_tensors(make_fake_batch())
        #             .repeat()
        #             .prefetch(tf.data.AUTOTUNE))
        # =====================================================

        #########################
        #   Loading the Model   #
        #########################
        # TODO: make an internal function

        # compile the model
        logging.info('Loading Model...')
        input_shape = [1, img_height, img_width, 3]

        keras_model = parse_model_name(model_name, img_height, img_width, num_classes)
    
        # build the layers
        keras_model(tf.zeros(input_shape))
        
        # compile the model with learning rates and optimizers
        keras_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
            loss=losses.CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing), # Label smoothing
            metrics=[metrics.CategoricalAccuracy()],
        )

        logging.info('Finished Loading Model!')


    #################
    #   Callbacks   #
    #################
    callbacks = get_callbacks(keras_model_dir, model+'_checkpoint.keras', stopping_threshold)

    ################
    #   Training   #
    ################

    # fit the model with custom callbacks and the datasets we created
    logging.info('Starting training...')
    keras_model.fit(train_ds,
                    epochs=1000000000,
                    validation_data=val_ds, 
                    callbacks=callbacks,
                    # validation_steps=10,
                    # steps_per_epoch=2000,
                    )

    keras_model_path = os.path.join(keras_model_dir, model+'.keras')
    keras_model.save(keras_model_path)

