import pandas as pd 
import os
import logging
import math

import tensorflow as tf
from tensorflow.keras import layers

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir, get_images_dir, get_record_path

from utils.file_handler.pickle import load_ids # TODO change from master to m0 or something

# TODO use the config file to get these variables
IMG_WIDTH=313
IMG_HEIGHT=437

IMG_EXTS=['.jpg']

###########################################################
#   preprocessing the images to be stored in a TFRecord   #
###########################################################

@tf.function
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    # image = tf.image.decode_image(image, channels=3) # decode_jpg or decode_png
    # image = tf.image.decode_jpeg(image)

    image = tf.io.decode_jpeg(
        image ,
        channels=3,
        fancy_upscaling=False,
        dct_method='INTEGER_FAST'
    )

    image.set_shape([None, None, 3]) # do I need this?
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH]) 

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


################
#   datasets   #
################

def build_dataset(paths, labels, unique_labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    ds = ds.apply(tf.data.Dataset.ignore_errors)

    return ds

def resolve_path(img_dir: str, file_id: str) -> str | None:
    '''
    Return the first existing file matching <file_id> plus one of the accepted extensions inside <img_dir>.
    If nothing exists, return None.
    '''

    for ext in IMG_EXTS:
        canidate = os.path.join(img_dir, file_id + ext)
    
        if os.path.isfile(canidate):
            return canidate
    return None

def generate_datasets(pl: PLS):
    '''
    Saves the dataset to disk.

    Args:
    Returns:
    '''

    _ids = load_ids(pl, 'm0', 'rb') 

    id_to_label = {id_: i for i, id_ in enumerate(_ids)}

    df = pd.DataFrame({'_id': _ids})
    df['label'] = df['_id'].map(id_to_label)

    img_dir = get_images_dir(pl)
    df['path'] = df['_id'].apply(lambda x: resolve_path(img_dir, x))

    df_present = df[df['path'].notna()].reset_index(drop=True)
    df_missing = df[df['path'].isna()].reset_index(drop=True)

    print("Total classes:", len(_ids))
    print("Present images:", len(df_present))
    print("Labels present in data:", df_present['label'].nunique())


    # make note of the missing ids for later
    missing_out = os.path.join(get_data_dir(), pl.value, f'missing_{pl.value}.csv')
    df_missing.to_csv(missing_out, index=False)
    logging.warning(
            'generate_datasets[%s]: %d images are absent; '
            'their IDs are saved to %s',
            pl.name, len(df_missing), missing_out
            )

    paths = tf.convert_to_tensor(df_present['path'], dtype=tf.string)
    labels = tf.convert_to_tensor(df_present['label'], dtype=tf.int32)
    
    print("_Ids Total classes:", len(_ids))
    print("Labels Total classes:", len(labels))

    ds = build_dataset(paths, labels, len(_ids))

    save_record(get_record_path(pl), ds)


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

def save_record(tfrecord_path, dataset):
    count = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, label in dataset:
            try:
                serialized = serialize_example(image, label)
                writer.write(serialized)
                count += 1
            except Exception as e:
                logging.error("Example %d failed: %s", label, e)
                continue
    logging.info('Wrote %d examples to %s', count, tfrecord_path)

def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.parse_tensor(parsed_example['image'], out_type=tf.float32)
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])  # use the same size used in preprocessing
    label = parsed_example['label']
    return image, label

def load_record(tfrecord_path, batch_size, shuffle, multiply, num_classes):
    ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE) # IMPORTANT TO PREVENT CPU READS FROM BEING SLOW
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if multiply > 1:
        ds = ds.repeat(multiply)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))
    # ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds 


