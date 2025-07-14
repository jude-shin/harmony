import pandas as pd 
import os
import logging

import tensorflow as tf

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir, get_val_dataset_dir, get_train_dataset_dir
from utils.file_handler.pickle import load_ids 

WIDTH=413
HEIGHT=312

###########################################################
#   preprocessing the images to be stored in a TFRecord   #
###########################################################

@tf.function
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # TODO: resize this to 413 by 312 or whatever the large one is
    # should this be a constant that we pull from the .env?
    image = tf.image.resize(image, [WIDTH, HEIGHT]) 

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
        '_ids': _ids # NOTE: if this is a 
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

    val_ds = load_dataset("val_ds.tfrecord", batch_size=32, shuffle=False, augment=False, multiply=1)
    
    train_ds = load_dataset("train_ds.tfrecord", batch_size=32, shuffle=True, augment=True, multiply=10)
    
    # TODO : save the datasets in the os.getenv('VAL_DATASET_PATH'), and os.getenv('TRAIN_DATASET_PATH')
    save_dataset(val_ds, get_val_dataset_dir)
    save_dataset(train_ds, get_train_dataset_dir)

    return train_ds, val_ds


###############
#   records   #
###############

def serialize_example(image, label):
    feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
            }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.parse_tensor(parsed_example['image'], out_type=tf.float32)
    image = tf.reshape(image, [WIDTH, HEIGHT, 3])  # use the same size used in preprocessing
    label = parsed_example['label']
    return image, label

def save_dataset(dataset, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, label in dataset:
            for i in range(image.shape[0]):  # image and label are batched
                serialized = serialize_example(image[i], label[i])
                writer.write(serialized)

def load_dataset(tfrecord_path, batch_size=32, shuffle=False, augment=False, multiply=1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and multiply > 1:
        # Repeat and augment each image N times
        def expand(image, label):
            images = [augment(image, label)[0] for _ in range(multiply)]
            labels = [label for _ in range(multiply)]
            return tf.data.Dataset.from_tensor_slices((images, labels))

        dataset = parsed_dataset.flat_map(expand)
    elif augment and multiply == 1:
        dataset = parsed_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = parsed_dataset

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

