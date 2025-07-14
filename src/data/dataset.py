import pandas as pd 
import os
import logging

import tensorflow as tf

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir
from utils.file_handler.pickle import load_ids 

RANDOM_RANGE = 10000

###########################################################
#   preprocessing the images to be stored in a TFRecord   #
###########################################################

@tf.function
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # TODO: resize this to 413 by 312 or whatever the large one is
    # should this be a constant that we pull from the .env?
    image = tf.image.resize(image, [224, 224]) 

    image = tf.image.convert_image_dtype(image, tf.float32)  

    return image, label


###################################
#   augmentation (during training)#
###################################

# note: this will be done to the training dataset real time
# prevents overfitting, stochastic, and decreases disk space

@tf.function
def augment_zoom_rotate(image, label):
    return image, label

@tf.function
def augment_blur(image, label):
    return image, label

@tf.function
def augment_saturation(image, label):
    image = tf.image.stateless_random_saturation(image, 0.5, 1, (1, 2))
    return image, label

@tf.function
def augment_contrast(image, label):
    return image, label

@tf.function
def augment_sharpness(image, label):
    return image, label

# other options for composing all of the augmentations 
@tf.function
def augment(image, label):
    fns = [augment_zoom_rotate, augment_blur, augment_saturation, augment_contrast, augment_sharpness]

    for fn in fns:
        image, label = fn(image, label)

    return image, label

################
#   datasets   #
################

def get_train_dataset(paths, labels, augment_factor=10, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Repeat each image N times
    ds = ds.flat_map(
            lambda path, label: tf.data.Dataset.from_tensors((path, label)).repeat(augment_factor)
            )

    # Stream, decode, augment
    # preprocessing happens in the model
    # any sized image (and I guess any type of image) can be tossed in
    # the preprocessing will resize and normalize the image 
    # BUT we do still need this to be a tensor 

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE) 
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(RANDOM_RANGE) # 1000
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_val_dataset(paths, labels, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # no augmentation should occur in the validation dataset
    # overfitting may occur 

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def generate_datasets(pl: PLS):
    '''
    Generates a training and validataion dataset.

    Saves the dataset as a df.pkl?

    Args:
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: a tuple containing:
            - a training dataset
            - a validation dataset
    '''
    
    data_dir = get_data_dir()

    img_dir = os.path.join(data_dir, pl.value, 'images')

    _ids = load_ids(pl, 'master', 'rb')
    
    df = pd.DataFrame({
        'label': range(0, len(_ids)),
        '_ids': _ids
        })

    # Stratified validation: ensure at least 1 sample per class
    val_df = df.groupby('label').sample(n=1, random_state=42)
    train_df = df.drop(val_df.index)
    
    # TODO: remove the hard coded '.jpg'
    # Create file paths
    train_paths = tf.convert_to_tensor(
            [os.path.join(img_dir, f+'.jpg') for f in train_df['_ids']],
            dtype=tf.string
            )
    train_labels = tf.convert_to_tensor(
            train_df['label'].values,
            dtype=tf.int32
            )

    val_paths = tf.convert_to_tensor(
            [os.path.join(img_dir, f+'.jpg') for f in val_df['_ids']],
            dtype=tf.string
            )
    val_labels = tf.convert_to_tensor(
            val_df['label'].values,
            dtype=tf.int32
            )

    train_ds = get_train_dataset(train_paths, train_labels, augment_factor=10)
    val_ds = get_val_dataset(val_paths, val_labels)

    # TODO : save the datasets in the os.getenv('VAL_DATASET_PATH'), and os.getenv('TRAIN_DATASET_PATH')
    logging.info('FINISHED GENERATING DATASETS')

    return train_ds, val_ds
    # model.fit(train_ds, validation_data=val_ds, epochs=10)
