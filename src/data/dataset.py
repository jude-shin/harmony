import pandas as pd 
import os
import logging

import tensorflow as tf

from utils.product_lines import PRODUCTLINES as PLS
from utils.file_handler.dir import get_train_dataset_path, get_val_dataset_path, get_data_dir, get_images_dir

from utils.file_handler.pickle import load_ids 

IMG_WIDTH=413
IMG_HEIGHT=312
IMG_EXTS=('.jpg', '.png')

###########################################################
#   preprocessing the images to be stored in a TFRecord   #
###########################################################

@tf.function
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

def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

def resolve_path(img_dir: str, file_id: str) -> str | None:
    '''
    Return the first existing file matching <file_id> plus one of the accepted extensions inside <img_dir>.
    If nothing exists, return None.
    '''
    # NOTE: we might be able to just use .jpg
    for ext in IMG_EXTS:
        canidate = os.path.join(img_dir, file_id + ext)
        if os.path.isfile(canidate):
            return canidate
    return None


def process_df(pl: PLS, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # construct the name 
    img_dir = get_images_dir(pl)

    df['path'] = df['_ids'].apply(lambda x: resolve_path(img_dir, x))

    present = df.dropna(subset=['path']).reset_index(drop=True)
    missing = df[df['path'].isna()].reset_index(drop=True)
    return present, missing

def generate_datasets(pl: PLS):
    '''
    Generates a training and validataion dataset.

    Saves the dataset

    Args:
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: a tuple containing:
            - a training dataset
            - a validation dataset
    '''

    _ids = load_ids(pl, 'master', 'rb') 

    df = pd.DataFrame({
        'label': range(0, len(_ids)),
        '_ids': _ids,
        })

    # Stratified validation: ensure at least 1 sample per class
    val_df = df.groupby('label', group_keys=False).sample(n=1, random_state=42)
    train_df = df.drop(val_df.index) # drops the index 'rows' labels by default
    

    train_df_present, train_df_missing = process_df(pl, train_df)
    val_df_present, val_df_missing = process_df(pl, val_df)

    # make note of the missing ids for later
    missing_out = os.path.join(get_data_dir(), pl.value, f'missing_{pl.value}.csv')
    pd.concat([train_df_missing, val_df_missing]).to_csv(missing_out, index=False)
    logging.warning(
            "generate_datasets[%s]: %d training and %d validation images are absent; "
            "their IDs are saved to %s",
            pl.name, len(train_df_missing), len(val_df_missing), missing_out
            )

    # Create file paths
    train_paths = tf.convert_to_tensor(train_df_present['path'], dtype=tf.string)
    train_labels = tf.convert_to_tensor(train_df_present['label'].values, dtype=tf.int32)

    val_paths = tf.convert_to_tensor(val_df_present['path'], dtype=tf.string)
    val_labels = tf.convert_to_tensor(val_df_present['label'].values, dtype=tf.int32)

    train_ds = build_dataset(train_paths, train_labels)
    val_ds = build_dataset(val_paths, val_labels)

    save_records(get_val_dataset_path(pl), val_ds)
    save_records(get_train_dataset_path(pl), train_ds)

    # val_ds = load_records("val_ds.tfrecord", batch_size=32, shuffle=False, augment=False, multiply=1)
    # 
    # train_ds = load_records("train_ds.tfrecord", batch_size=32, shuffle=True, augment=True, multiply=10)

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
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])  # use the same size used in preprocessing
    label = parsed_example['label']
    return image, label

def save_records(tfrecord_path, dataset):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, label in dataset:
            serialized = serialize_example(image, label)
            writer.write(serialized)

def load_records(tfrecord_path, batch_size=32, shuffle=False, augment=False, multiply=1):
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

