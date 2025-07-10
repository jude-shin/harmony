import pandas as pd 
import os
import logging

import tensorflow as tf
# import tensorflow_addons as tfa
import keras_cv

# from tensorflow import keras, models, layers
# from sklearn.model_selection import train_test_split

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir
from utils.file_handler.pickle import load_ids 

# ========================================

@tf.function
def augment_zoom_rotate(image, label):

    return image, label

@tf.function
def augment_blur(image, label):

    return image, label

@tf.function
def augment_saturation(image, label):
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
    # fns = [augment_zoom_rotate, augment_blur, augment_saturation, augment_contrast, augment_sharpness]
    fns = [augment_saturation, augment_contrast]
    for fn in fns:
        apply = tf.random.uniform([]) > 0.5
        image, label = tf.cond(apply, lambda: fn(image, label), lambda: (image, label))
    return image, label


# ========================================

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

    # ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE) 
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_val_dataset(paths, labels, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # no preprocessing should occur in the validation dataset
    # overfitting may occur 
    # ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ========================================
def generate_datasets(pl: PLS):
    '''
    Generates a training and validataion dataset.

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
    train_paths = [os.path.join(img_dir, f+'.jpg') for f in train_df['_ids']]
    train_labels = train_df['label'].values
    
    val_paths = [os.path.join(img_dir, f+'.jpg') for f in val_df['_ids']]
    val_labels = val_df['label'].values
    
    train_ds = get_train_dataset(train_paths, train_labels, augment_factor=10)
    val_ds = get_val_dataset(val_paths, val_labels)

    # TODO : save the datasets in the os.getenv('VAL_DATASET_PATH'), and os.getenv('TRAIN_DATASET_PATH')
    logging.info('FINISHED GENERATING DATASETS')

    return train_ds, val_ds
    # model.fit(train_ds, validation_data=val_ds, epochs=10)

