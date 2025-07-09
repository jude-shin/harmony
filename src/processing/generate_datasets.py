import pandas as pd 
import os

import tensorflow as tf
# import tensorflow_addons as tfa
import keras_cv

# from tensorflow import keras, models, layers
from sklearn.model_selection import train_test_split

# ========================================

@tf.function
def augment_zoom_rotate(image, label):
    # Random zoom scale
    scale = tf.random.uniform([], 0.95, 1.05)
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    new_h = tf.cast(scale * tf.cast(h, tf.float32), tf.int32)
    new_w = tf.cast(scale * tf.cast(w, tf.float32), tf.int32)

    # Resize to zoom
    image = tf.image.resize(image, [new_h, new_w])

    # Pad back to original size with random color
    pad_color = tf.random.uniform([3], 0.0, 1.0)
    pad_color = tf.reshape(pad_color, [1, 1, 3])
    pad_color = tf.tile(pad_color, [h, w, 1])

    image = tf.image.resize_with_crop_or_pad(image, h, w)
    image = tf.where(tf.equal(image, 0.0), pad_color, image)

    # Apply rotation using KerasCV
    image = keras_cv.layers.RandomRotation(
            factor=0.028,  # ~5 degrees = 5/180 ≈ 0.028
            fill_mode="reflect"
            )

    return image, label

@tf.function
def augment_blur(image, label):
    # image = tfa.image.gaussian_filter2d(image, filter_shape=(5, 5), sigma=1.0)
    image = keras_cv.layers.GaussianBlur(kernel_size=5, sigma=1.0)(image)
    return image, label

@tf.function
def augment_saturation(image, label):
    saturation = tf.random.shuffle([0.5, 1.5])[0]
    image = tf.image.adjust_saturation(image, saturation)
    return image, label

@tf.function
def augment_contrast(image, label):
    contrast = tf.random.shuffle([0.5, 1.5])[0]
    image = tf.image.adjust_contrast(image, contrast)
    return image, label

@tf.function
def augment_sharpness(image, label):
    # blur = tfa.image.gaussian_filter2d(image, filter_shape=(3, 3), sigma=1.0)
    blur = keras_cv.layers.GaussianBlur(kernel_size=5, sigma=1.0)(image)
    sharpness = tf.random.shuffle([0.5, 1.5])[0]
    image = tf.clip_by_value(image * sharpness + blur * (1 - sharpness), 0.0, 1.0)
    return image, label

# other options for composing all of the augmentations 
@tf.function
def augment(image, label):
    fns = [augment_zoom_rotate, augment_blur, augment_saturation, augment_contrast, augment_contrast]
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
def generate_datasets(csv_path, img_dir):
    '''
    Generates a training and validataion dataset.

    Args:
        csv_path (str): the path to the csv in the format of "filename, label"
        img_dir (str): the directory that the images are stored in
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: a tuple containing:
            - a training dataset
            - a validation dataset
    '''

    df = pd.read_csv(csv_path)
    
    # Stratified validation: ensure at least 1 sample per class
    val_df = df.groupby("label").sample(n=1, random_state=42)
    train_df = df.drop(val_df.index)
    
    # Create file paths
    train_paths = [os.path.join(img_dir, f) for f in train_df['filename']]
    train_labels = train_df['label'].values
    
    val_paths = [os.path.join(img_dir, f) for f in val_df['filename']]
    val_labels = val_df['label'].values
    
    train_ds = get_train_dataset(train_paths, train_labels, augment_factor=10)
    val_ds = get_val_dataset(val_paths, val_labels)

    # TODO : save the datasets in the os.getenv('VAL_DATASET_PATH'), and os.getenv('TRAIN_DATASET_PATH')

    return train_ds, val_ds
    # model.fit(train_ds, validation_data=val_ds, epochs=10)

