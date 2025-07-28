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
    image = tf.image.decode_image(image, channels=3) # decode_jpg or decode_png
    image.set_shape([None, None, 3]) 
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH]) 

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


######################################
#   augmentation (during training)   #
######################################

# note: this will be done to the training dataset real time
# prevents overfitting, stochastic, and decreases disk space
#################
#   GEOMETRIC   #
################# 

@tf.function
def augment_skew(image, label):
    max_skew = 0.03
    
    skew_x = tf.random.uniform([], -max_skew, max_skew)
    skew_y = tf.random.uniform([], -max_skew, max_skew)

    transform = [1.0, skew_x, 0.0,
                 skew_y, 1.0, 0.0,
                 0.0,    0.0]
    transform = tf.convert_to_tensor([transform], dtype=tf.float32)  # batch of 1 transform

    image = tf.expand_dims(image, 0)
    output_shape = tf.shape(image)[1:3]

    image_skewed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=output_shape,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0
    )

    image_skewed = tf.squeeze(image_skewed, 0)
    image_skewed = tf.cast(image_skewed, image.dtype)
    return image_skewed, label


@tf.function
def augment_rotation(image, label):
    angle_rad = tf.random.uniform([], -math.pi, math.pi)

    # Image dimensions
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    center = [width_f / 2, height_f / 2]

    # Rotation matrix
    cos_angle = tf.cos(angle_rad)
    sin_angle = tf.sin(angle_rad)
    rotation_matrix = tf.reshape(
        [cos_angle, -sin_angle, (1 - cos_angle) * center[0] + sin_angle * center[1],
         sin_angle, cos_angle, (1 - cos_angle) * center[1] - sin_angle * center[0],
         0, 0],
        [8]
    )

    # Rotate using ImageProjectiveTransformV3
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, axis=0),
        transforms=tf.expand_dims(rotation_matrix, axis=0),
        output_shape=[height, width],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0
    )

    rotated_image = tf.squeeze(rotated, axis=0)

    return rotated_image, label

# @tf.function
# def augment_translation(image, label):
#     image = tf.expand_dims(image, axis=0)
#     translated = layers.RandomTranslation(
#         height_factor=0.2,
#         width_factor=0.2,
#         fill_mode='constant',
#         fill_value=0.0,
#         interpolation='bilinear'
#     )(image, training=True)
#     return tf.squeeze(translated, axis=0), label


@tf.function
def augment_translation(image, label):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height_f, width_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    # Random translation factors (-20% to 20%)
    translate_x = tf.random.uniform([], -0.2, 0.2) * width_f
    translate_y = tf.random.uniform([], -0.2, 0.2) * height_f

    # Translation transform
    transform = [1.0, 0.0, -translate_x,
                 0.0, 1.0, -translate_y,
                 0.0, 0.0]

    # Apply transformation
    translated = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=[transform],
        output_shape=[height, width],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0
    )

    translated_image = tf.squeeze(translated, axis=0)

    return translated_image, label




#####################
#   NON-GEOMETRIC   # 
#####################

@tf.function
def augment_blur(image, label):
    kernel_vals = [
        [1.,  4.,  6.,  4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1.,  4.,  6.,  4., 1.]
    ]

    kernel = tf.constant(kernel_vals, dtype=tf.float32)
    kernel = kernel / tf.reduce_sum(kernel)
    # kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.reshape(kernel, [5, 5, 1, 1])
    kernel = tf.tile(kernel, [1, 1, tf.shape(image)[-1], 1])

    # Prepare image for convolution
    tf_img = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
    image = tf.nn.depthwise_conv2d(tf_img, kernel, strides=[1,1,1,1], padding='SAME')
    image = tf.squeeze(image, axis=0)
    image = tf.cast(image, image.dtype)
    
    return image, label

@tf.function
def augment_saturation(image, label):
    image = tf.image.random_saturation(image, 0.5, 1)
    return image, label

@tf.function
def augment_contrast(image, label):
    image = tf.image.random_contrast(image, 0.2, 0.5)
    return image, label

# @tf.function
# def augment_hue(image, label):
# # NOTE: I don't think that it is beneficial to change the color of the image. this might hurt us
#     image = tf.image.random_hue(image, 0.1)
#     return image, label

@tf.function
def augment_brightness(image, label):
    image = tf.image.random_brightness(image, 0.2)
    return image, label


@tf.function
def augment_non_geometric(image, label):
    fns = [
        augment_blur,
        augment_saturation,
        augment_contrast,
        augment_brightness,
    ]

    for fn in fns:
        apply = tf.random.uniform(()) < 0.5
        image, label = tf.cond(apply, lambda: fn(image, label), lambda: (image, label))

    return image, label

@tf.function
def augment_geometric(image, label):
    i = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    return tf.case([
        (tf.equal(i, 0), lambda: augment_rotation(image, label)),
        (tf.equal(i, 1), lambda: augment_skew(image, label)),
        (tf.equal(i, 2), lambda: augment_translation(image, label)),
    ])



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


# def process_df(pl: PLS, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     img_dir = get_images_dir(pl) # this does not work
#     # img_dir = os.path.join(get_data_dir(), pl.value, 'images')
# 
#     df['path'] = df['_id'].apply(lambda x: resolve_path(img_dir, x))
# 
#     present = df[df['path'].notna()].reset_index(drop=True)
#     missing = df[df['path'].isna()].reset_index(drop=True)
# 
# 
#     logging.info('Number Present: %d', len(present))
#     logging.info('Number Missing: %d', len(missing))
# 
#     return present, missing


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

def load_record(tfrecord_path, batch_size, shuffle, augment, multiply, num_classes):
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    parsed_ds = raw_ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        parsed_ds = parsed_ds.map(augment_geometric, num_parallel_calls=tf.data.AUTOTUNE) 
        parsed_ds = parsed_ds.map(augment_non_geometric, num_parallel_calls=tf.data.AUTOTUNE)

    if multiply > 1:
        parsed_ds = parsed_ds.repeat(multiply)

    if shuffle:
        parsed_ds = parsed_ds.shuffle(buffer_size=1000)

    ds = parsed_ds.batch(batch_size)

    # one hot encode the labels for smooth labels
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=num_classes)))

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds 

