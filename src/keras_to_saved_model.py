import logging
import os

import tensorflow as tf

from tensorflow.keras import models

from cnn.model_structure import *


def keras_to_saved_model(source: str, target: str): 
    # load the tf keras model
    model = models.load_model(source) 

    height, width = model.preprocess.target_size
    logging.info('height: %d', height)
    logging.info('width : %d', width)

    dummy = tf.zeros([1, height, width, 3], dtype=tf.float32)
    _ = model(dummy, training=False)

    # model.build((1, height, width, 3))

    # # save it with the 
    model.export(target, format='tf_saved_model')


# keras_to_saved_model('/home/jude/m0.keras', '/home/jude/harmony/saved_models/save_lorcana/m0/2/')
keras_to_saved_model('/home/jude/harmony/keras_models/lorcana/2025.07.25_21.02.43/m0/m0.keras', '/home/jude/harmony/saved_models/save_lorcana/m0/2/')

# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m0.keras', '/home/jude/harmony/saved_models/save_pokemon/m0/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m1.keras', '/home/jude/harmony/saved_models/save_pokemon/m1/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m2.keras', '/home/jude/harmony/saved_models/save_pokemon/m2/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m3.keras', '/home/jude/harmony/saved_models/save_pokemon/m3/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m4.keras', '/home/jude/harmony/saved_models/save_pokemon/m4/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m5.keras', '/home/jude/harmony/saved_models/save_pokemon/m5/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m6.keras', '/home/jude/harmony/saved_models/save_pokemon/m6/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m7.keras', '/home/jude/harmony/saved_models/save_pokemon/m7/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m8.keras', '/home/jude/harmony/saved_models/save_pokemon/m8/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m9.keras', '/home/jude/harmony/saved_models/save_pokemon/m9/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m10.keras', '/home/jude/harmony/saved_models/save_pokemon/m10/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m11.keras', '/home/jude/harmony/saved_models/save_pokemon/m11/1/')
# keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m12.keras', '/home/jude/harmony/saved_models/save_pokemon/m12/1/')

