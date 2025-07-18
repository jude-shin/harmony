import logging
import time 
import gc

# import matplotlib
# matplotlib.use('TkAgg')  
# import matplotlib.pyplot as plt

import tensorflow as tf 

from tensorflow import keras

from data.collect import download_images_parallel, collect
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 
from data.dataset import generate_datasets
from training.train import train 

from data.dataset import augment_blur, augment_saturation, augment_contrast, augment_hue, augment_brightness, augment_rotation, augment_zoom_rotate, augment_all

logging.getLogger().setLevel(10)

# def visualize(img_path):
#     label = 0 # dummy label
#     
#     image_raw = tf.io.read_file(img_path)
#     image = tf.image.decode_jpeg(image_raw, channels=3)
#     # image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     
#     augmentations = [
#         ('Original', lambda img, lbl: (img, lbl)),
#         ('Blur', augment_blur),
#         ('Saturation', augment_saturation),
#         ('Contrast', augment_contrast),
#         ('Hue', augment_hue),
#         ('Brightness', augment_brightness),
#         ('Rotation', augment_rotation),
#         ('Zoom+Rotate', augment_zoom_rotate),
#         ('All Combined', augment_all),
#     ]
#     
#     fig, axs = plt.subplots(2, (len(augmentations) + 1) // 2, figsize=(18, 8))
#     axs = axs.flatten()
#     
#     for idx, (name, fn) in enumerate(augmentations):
#         aug_img, _ = fn(image, label)
#         axs[idx].imshow(tf.clip_by_value(aug_img, 0, 1))
#         axs[idx].set_title(name)
#         axs[idx].axis('off')
#     
#     plt.tight_layout()
#     plt.savefig("augmentation_results.png")
#     # plt.show()




if __name__ == '__main__':
    # clear tensorflow
    keras.backend.clear_session()

    # force garbage collection
    gc.collect()

    # expand the gpus for growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # ----------------------------------
    # st = time.time()
    # collect(PLS.LORCANA)
    generate_datasets(PLS.LORCANA)
    # logging.warning(' ----> ELAPSED TIME: ' + get_elapsed_time(st))
    # ----------------------------------
    
    train(PLS.LORCANA)

    # ----------------------------------
    # path = '/home/storepass/harmony/src/Selection_001.png'
    # visualize(path)
    # ----------------------------------



