import logging
import time 
import gc

# import matplotlib
# matplotlib.use('TkAgg')  
# import matplotlib.pyplot as plt

import tensorflow as tf 

from tensorflow import keras
from tensorflow.keras import mixed_precision 

from data.collect import download_images_parallel, collect
from utils.product_lines import PRODUCTLINES as PLS
from utils.time import get_elapsed_time 
from data.dataset import generate_datasets
from training.train import train_product_line 

from data.dataset import * 

logging.getLogger().setLevel(10)

# def visualize(img_path):
#     label = 0 # dummy label
#     
#     image_raw = tf.io.read_file(img_path)
#     image = tf.image.decode_image(image_raw, channels=3)
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

    # expand all gpus for growth
    # TODO: for some reason after putting this in a docker container, you can't change the devices after they have been initalized or something
    # gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    
    # distribute the workload across all gpus
    strategy = tf.distribute.MirroredStrategy()
    
    # makes things faster?
    mixed_precision.set_global_policy("mixed_float16")

    # DOWNLOAD THE IMAGES
    # collect(PLS.LORCANA)
    
    with strategy.scope():
        # GENERATE THE DATASETS
        generate_datasets(PLS.LORCANA)
        
        # TRAIN THE MODEL
        train_product_line(PLS.LORCANA, ['m0'])

    # ----------------------------------
    # path = '/home/storepass/harmony/src/Selection_001.png'
    # visualize(path)
    # ----------------------------------



