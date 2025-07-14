import tensorflow as tf
import os

from data.dataset import augment

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
    image = tf.reshape(image, [224, 224, 3])  # use the same size used in preprocessing
    label = parsed_example['label']
    return image, label



def write_tfrecord(dataset, tfrecord_path):
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
