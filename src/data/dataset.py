import pandas as pd 
import os
import logging

import random

import tensorflow as tf

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_data_dir, get_images_dir, get_record_path

from utils.file_handler.pickle import load_ids 

IMG_WIDTH=313
IMG_HEIGHT=437
IMG_EXTS=['.jpg']

###########################################################
#   preprocessing the images to be stored in a TFRecord   #
###########################################################

# @tf.function
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # TODO: resize this to 413 by 312 or whatever the large one is
    # should this be a constant that we pull from the .env?
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH]) 

    image = tf.image.convert_image_dtype(image, tf.float32)  

    return image, label


######################################
#   augmentation (during training)   #
######################################

# note: this will be done to the training dataset real time
# prevents overfitting, stochastic, and decreases disk space

# @tf.function
def augment_zoom_rotate(image, label):
    # TODO
    # slightly shrink the image and rotate it within the original bounds
    # this might not be needed
    return image, label

# @tf.function
def augment_skew(image, label, max_skew=0.3):
    # max_skew: maximum shearing factor, e.g., 0.3 = up to 30% skew
    
    # Randomly pick horizontal and vertical skew values
    skew_x = tf.random.uniform([], -max_skew, max_skew)
    skew_y = tf.random.uniform([], -max_skew, max_skew)
    
    # Affine transform matrix for skewing (shearing)
    # [1, skew_x, 0,
    #  skew_y, 1, 0,
    #  0, 0]
    transform = [1.0, skew_x, 0.0,
                 skew_y, 1.0, 0.0,
                 0.0,    0.0]
    # The transform is a flat list, as required by TensorFlow

    # Make batch dimension
    image = tf.expand_dims(image, 0)

    # Use 'BILINEAR' for smooth skew, 'REFLECT' to fill empty areas
    image_skewed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=[transform],
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )

    image_skewed = tf.squeeze(image_skewed, 0)
    image_skewed = tf.cast(image_skewed, image.dtype)
    return image_skewed, label


# @tf.function
def augment_rotation(image, label):
    if random.choice([True, False]):
        image = tf.image.flip_up_down(image)
        image = tf.image.flip_left_right(image)
    return image, label

# @tf.function
def augment_blur(image, label):
    for _ in range(random.randint(1, 3)): # this 
        # 3x3 Gaussian kernel, sigma=1
        # kernel_vals = [[1., 2., 1.],
        #                [2., 4., 2.],
        #                [1., 2., 1.]]

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
        img = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
        blurred = tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding='SAME')
        blurred = tf.squeeze(blurred, axis=0)
        blurred = tf.cast(blurred, image.dtype)
        
        # Randomly apply blur
        do_blur = tf.random.uniform([]) > 0.5
        image = tf.cond(do_blur, lambda: blurred, lambda: image)
    return image, label

# @tf.function
def augment_saturation(image, label):
    image = tf.image.random_saturation(image, 0.5, 1)
    return image, label

# @tf.function
def augment_contrast(image, label):
    image = tf.image.random_contrast(image, 0.2, 0.5)
    return image, label

# @tf.function
# NOTE: I don't think that it is beneficial to change the color of the image. this might hurt us
def augment_hue(image, label):
    image = tf.image.random_hue(image, 0.1)
    return image, label

# @tf.function
def augment_brightness(image, label):
    image = tf.image.random_brightness(image, 0.2)
    return image, label

# other options for composing all of the augmentations 
# @tf.function.
def augment_all(image, label):
    # fns = [augment_zoom_rotate, augment_blur, augment_saturation, augment_contrast, augment_hue, augment_brightness, augment_rotation]
    fns = [augment_blur, augment_saturation, augment_contrast, augment_brightness, augment_rotation, augment_skew]

    for fn in fns:
        image, label = fn(image, label)

    return image, label

################
#   datasets   #
################

def build_dataset(paths, labels):
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


def process_df(pl: PLS, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    img_dir = get_images_dir(pl) # this does not work
    # img_dir = os.path.join(get_data_dir(), pl.value, 'images')

    df['path'] = df['_id'].apply(lambda x: resolve_path(img_dir, x))

    present = df[df['path'].notna()].reset_index(drop=True)
    missing = df[df['path'].isna()].reset_index(drop=True)


    logging.info('Number Present: %d', len(present))
    logging.info('Number Missing: %d', len(missing))

    return present, missing


def generate_datasets(pl: PLS):
    '''
    Saves the dataset to disk.

    Args:
    Returns:
    '''

    _ids = load_ids(pl, 'master', 'rb') 

    df = pd.DataFrame({
        'label': range(0, len(_ids)),
        '_id': _ids,
        })

    # resolve missing images
    df_present, df_missing = process_df(pl, df)

    # ========================================== 
    # make note of the missing ids for later
    missing_out = os.path.join(get_data_dir(), pl.value, f'missing_{pl.value}.csv')
    # pd.concat([train_df_missing, val_df_missing]).to_csv(missing_out, index=False)
    df_missing.to_csv(missing_out, index=False)
    logging.warning(
            'generate_datasets[%s]: %d images are absent; '
            'their IDs are saved to %s',
            pl.name, len(df_missing), missing_out
            )
    # ========================================== 

    # Create file paths
    paths = tf.convert_to_tensor(df_present['path'], dtype=tf.string)
    labels = tf.convert_to_tensor(df_present['label'].values, dtype=tf.int32)

    ds = build_dataset(paths, labels)
    
    save_record(get_record_path(pl), ds)

    # usage
    # val_ds = load_record('record.tfrecord', batch_size=32, shuffle=False, augment=False, multiply=1)
    # train_ds = load_record('record.tfrecord', batch_size=32, shuffle=True, augment=True, multiply=10)


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
                logging.error("Example %d failed: %s", n.numpy(), e)
                continue
    logging.info('Wrote %d examples to %s', count, tfrecord_path)

def load_record(tfrecord_path, batch_size=32, shuffle=False, augment=False, multiply=1):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if augment and multiply > 1:
        # Repeat and augment each image N times
        def expand(image, label):
            images = [augment_all(image, label)[0] for _ in range(multiply)]
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

