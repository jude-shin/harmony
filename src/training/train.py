import logging 
import os

import tensorflow as tf

from tensorflow.keras import optimizers, losses, metrics, models
from time import localtime, strftime

from data.dataset import load_record
from cnn.model_structure import * 
from utils.file_handler.dir import get_record_path, get_keras_model_dir
from utils.file_handler.toml import * 
from utils.product_lines import PRODUCTLINES as PLS
from training.callbacks import * 

# from cnn.sequential_models import model_classic_1, model_classic_15
def train_product_line(pl: PLS, models: list[str]):
    # load the config.toml based on the model and product line
    config_path = get_config_path(pl)
    config = load_config(config_path)

    if not models: # if the list is empty, default and train all of the models in the config
        models = list(config.keys())

    # TODO: make these run in parallel?
    for model in models:
        model_config = config[model]
        train_model(pl, model, model_config)

# def continue_training_product_line(pl: PLS, models: list[str], version: str):
#     for model in models:
#         # load the config that is stored in the appropriate directory
#         continue_training(pl, model, version)
    
# def continue_training(pl: PLS, model: str, version: str):
#     ###########################
#     #   keras_models verson   #
#     ###########################
# 
#     # get the dir name based on the name of the file (which was based on the time of creation)
#     keras_model_dir = os.path.join(get_keras_model_dir(), pl.value, version, model)
# 
# 
#     #############################
#     #   Load the Model Config   #
#     #############################
# 
#     config = load_config(keras_model_dir)
# 
#     #################
#     #   Variables   #
#     #################
# 
#     # Training Augmentation Multiplication
#     batch_size = config['batch_size']
#     # model_name = config['model_name']
#     # img_height = config['img_height']
#     # img_width = config['img_width']
#     num_classes = config['num_unique_classes']
#     augment_multiplication = config['augment_multiplication']
#     # learning_rate = config['learning_rate']
#     # beta_1 = config['beta_1']
#     # beta_2 = config['beta_2']
#     # label_smoothing = config['label_smoothing']
# 
#     # TODO
#     stopping_threshold = config['stopping_threshold']
# 
#     # Callbacks?
#     #   Stopping threshold (at 98 or so for val accuracy)
# 
# 
#     ############################
#     #   Loading the Datasets   #
#     ############################
#     # TODO: make an internal function
# 
#     # load the validation and training datasets from the record stored on disk
#     # training data should be multiplied more than the validation data
#     # training data should be shuffled and augmented
#     # validation can be augmented or shuffled
#     logging.info('Loading Training Dataset from TFRecord...')
#     train_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=True, multiply=augment_multiplication, num_classes=num_classes)
#     logging.info('Finished Loading Training Dataset!')
# 
#     logging.info('Loading Validation Dataset from TFRecord...')
#     val_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=False, multiply=1, num_classes=num_classes)
#     logging.info('Finished Loading Validation Dataset!')
# 
# 
#     #########################
#     #   Loading the Model   #
#     #########################
#     # TODO make an internal function
# 
#     # load the model that was saved in the directory
#     logging.info('Loading Model...')
#     keras_model_path = os.path.join(keras_model_dir, model+'.keras')
# 
#     if not os.path.exists(keras_model_path):
#         keras_model_path = os.path.join(keras_model_dir, model+'_checkpoint.keras')
# 
#     keras_model = models.load_model(keras_model_path)
#     logging.info('Finished Loading Model!')
# 
# 
#     #################
#     #   Callbacks   #
#     #################
# 
#     # defines when the model will stop training
#     accuracy_threshold_callback = EarlyStoppingByValThreshold(
#             monitor='val_categorical_accuracy',
#             threshold=stopping_threshold,
#             )
# 
#     # saves a snapshot of the model while it is training
#     checkpoint_path = os.path.join(keras_model_dir, model+"_checkpoint.keras")
#     checkpoint_callback = callbacks.ModelCheckpoint(
#         filepath=checkpoint_path, save_weights_only=False, save_best_only=True,
#         monitor='val_loss',
#         mode='min'
#     )
# 
#     # logs the epoch, accuracy, and loss for a training session
#     csv_path = os.path.join(keras_model_dir, "training_logs.csv")
#     csv_logger_callback = CsvLoggerCallback(csv_path)
# 
# 
#     ################
#     #   Training   #
#     ################
# 
#     # fit the model with custom callbacks and the datasets we created
#     logging.info('Starting training...')
#     keras_model.fit(train_ds,
#               epochs=1,
#               validation_data=val_ds, 
#               callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback]
#               )
# 
#     keras_model_path = os.path.join(keras_model_dir, model+'.keras')
#     keras_model.save(keras_model_path)



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
    # TODO: make an internal function

    # load the validation and training datasets from the record stored on disk
    # training data should be multiplied more than the validation data
    # training data should be shuffled and augmented
    # validation can be augmented or shuffled
    logging.info('Loading Training Dataset from TFRecord...')
    train_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=True, multiply=augment_multiplication, num_classes=num_classes)
    logging.info('Finished Loading Training Dataset!')

    logging.info('Loading Validation Dataset from TFRecord...')
    val_ds = load_record(get_record_path(pl), batch_size=batch_size, shuffle=False, multiply=1, num_classes=num_classes)
    logging.info('Finished Loading Validation Dataset!')
   

    #########################
    #   Loading the Model   #
    #########################
    # TODO: make an internal function

    # compile the model
    logging.info('Loading Model...')
    input_shape = [1, img_height, img_width, 3]

    logging.info(input_shape)


    
    # distribute the workload across ALL gpus
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
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

    # defines when the model will stop training
    accuracy_threshold_callback = EarlyStoppingByValThreshold(
            monitor='val_categorical_accuracy',
            threshold=stopping_threshold,
            )

    # saves a snapshot of the model while it is training
    checkpoint_path = os.path.join(keras_model_dir, model+"_checkpoint.keras")
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=False, save_best_only=True,
        monitor='val_loss',
        mode='min',
    )

    # logs the epoch, accuracy, and loss for a training session
    csv_path = os.path.join(keras_model_dir, "training_logs.csv")
    csv_logger_callback = CsvLoggerCallback(csv_path)
   

    ################
    #   Training   #
    ################

    # fit the model with custom callbacks and the datasets we created
    logging.info('Starting training...')
    keras_model.fit(train_ds,
              epochs=1000000000,
              validation_data=val_ds, 
              callbacks=[accuracy_threshold_callback, checkpoint_callback, csv_logger_callback]
              )

    keras_model_path = os.path.join(keras_model_dir, model+'.keras')
    keras_model.save(keras_model_path)

