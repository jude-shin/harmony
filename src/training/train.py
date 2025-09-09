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

        logging.info('Loading Training Dataset from TFRecord...')
        train_ds = load_record(pl=pl, batch_size=batch_size, shuffle=True, model='m0')
        logging.info('Finished Loading Training Dataset!')

        logging.info('Loading Validation Dataset from TFRecord...')
        val_ds = load_record(pl=pl, batch_size=batch_size, shuffle=False, model='m0')
        logging.info('Finished Loading Validation Dataset!')

        #########################
        #   Loading the Model   #
        #########################
        # NOTE: we are loading the embedding

        # compile the model
        logging.info('Loading ANN_CLASSIFIER...')


        config = load_model_config(pl)
        config = config['ann']
        num_unique_classes = config['num_unique_classes']

        embedder = build_embedder(pl, emb_dim=256)

        # simple classifier head (training only)
        ann_classifier_out = layers.Dense(num_unique_classes, dtype='float32', name='logits')(embedder.output)
        ann_classifier = models.Model(embedder.input, ann_classifier_out)
        ann_classifier.compile(
                optimizer=optimizers.Adam(0.001),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[metrics.SparseCategoricalAccuracy()]
                )

        logging.info('Finished Loading ANN_CLASSIFIER!')


    #################
    #   Callbacks   #
    #################
    callbacks = get_callbacks(keras_model_dir, model+'_checkpoint.keras', stopping_threshold)

    ################
    #   Training   #
    ################

    # fit the model with custom callbacks and the datasets we created
    logging.info('Starting training...')
    ann_classifier.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=10,
            )
    
    # TODO: change this to the tensorflow serving path
    keras_model_path = os.path.join(keras_model_dir, model)

    embedder = models.Model(ann_classifier.input, ann_classifier.get_layer('emb_l2').output, name='embedder_export') 
    embedder.save(keras_model_path) # this should be the path to a tensorflow serving


    

