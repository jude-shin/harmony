# NOTE: all of these things just don't work out because the augmentation layer which uses legacy code
# Make an inference only graph that is able to be exported easily to be used in tensorflow serving


# --- keras_to_saved_model.py -----------------------------------------------
import zipfile, pathlib, tempfile, tensorflow as tf
import os
from cnn.model_structure import PreprocessingLayer          # your resize layer
from tensorflow.keras import applications, layers, Model

from utils.file_handler.dir import get_keras_model_dir, get_saved_model_dir

# ARCHIVE = "/home/jude/harmony/keras_models/pokemon/m0.keras"
# EXPORT_DIR = "/home/jude/harmony/saved_models/pokemon/m0/1"

ARCHIVE = os.path.join(get_keras_model_dir(), 'pokemon', 'm0.keras')
EXPORT_DIR = os.path.join(get_saved_model_dir(), 'pokemon', 'm0', '1')

HEIGHT, WIDTH, NUM_CLASSES = 437, 313, 23675

# --------------------------------------------------------------------------- #
# 1. Build an inference-only network (no Random* layers)                      #
# --------------------------------------------------------------------------- #
inputs = layers.Input(shape=(HEIGHT, WIDTH, 3))
x = PreprocessingLayer(target_size=[HEIGHT, WIDTH])(inputs)

backbone = applications.ResNet152(
    include_top=False, weights=None, input_tensor=x)
x = layers.GlobalAveragePooling2D()(backbone.output)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

infer_net = Model(inputs, outputs, name="ResNet152_inference")
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 2. Extract the weights file embedded in the .keras archive and load it      #
# --------------------------------------------------------------------------- #
with zipfile.ZipFile(ARCHIVE) as z:
    weights_member = next(n for n in z.namelist() if n.endswith("weights.h5"))
    tmp = pathlib.Path(tempfile.mkdtemp()) / "weights.h5"
    z.extract(weights_member, tmp.parent)
    tmp = tmp.parent / weights_member.split("/")[-1]         # full path to .h5

infer_net.load_weights(tmp, skip_mismatch=True)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 3. Export a clean SavedModel for TensorFlow Serving                         #
# --------------------------------------------------------------------------- #
tf.saved_model.save(infer_net, EXPORT_DIR)
print("✓ SavedModel ready →", EXPORT_DIR)



## import zipfile, tempfile, pathlib, tensorflow as tf
## from cnn.model_structure import parse_model_name
## 
## ARCHIVE = "/home/jude/harmony/keras_models/pokemon/m0.keras"
## HEIGHT, WIDTH, NUM_CLASSES = 437, 313, 23675
## 
## # 1.  Build the graph you want to serve (no augmentation stack)
## net = parse_model_name("ResNet152", HEIGHT, WIDTH, NUM_CLASSES)
## 
## # 2.  Pull the HDF5 weights file out of the .keras archive
## with zipfile.ZipFile(ARCHIVE) as z:
##     wfile = next(n for n in z.namelist() if n.endswith("weights.h5"))
##     tmpdir = tempfile.mkdtemp()
##     weights_path = pathlib.Path(z.extract(wfile, tmpdir))
## 
## # 3.  Load weights
## net.load_weights(weights_path)
## 
## # 4.  Export to TensorFlow Serving format
## tf.saved_model.save(net, "/home/jude/harmony/saved_models/pokemon/m0/1")



# import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
# 
# import logging
# 
# import tensorflow as tf
# 
# import keras_cv 
# from tensorflow.keras import models
# 
# from cnn.model_structure import *
# 
# 
# def keras_to_saved_model(source: str, target: str): 
#     # load the tf keras model
#     model = models.load_model(
#             source, 
#             compile=False,
#             safe_mode=False,
#             )
# 
#     # # height, width = model.preprocess.target_size
#     # # logging.info('height: %d', height)
#     # # logging.info('width : %d', width)
# 
#     # # dummy = tf.zeros([1, height, width, 3], dtype=tf.float32)
#     # # _ = model(dummy, training=False)
# 
#     # model.build((1, height, width, 3))
# 
#     # # save it with the 
#     # model.export(target, format='tf_saved_model')
#     model.save(target)
# 
# 
# # keras_to_saved_model('/home/jude/m0.keras', '/home/jude/harmony/saved_models/save_lorcana/m0/2/')
# keras_to_saved_model('/home/jude/harmony/keras_models/pokemon/m0.keras', '/home/jude/harmony/saved_models/pokemon/m0/1/')
# 
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m0.keras', '/home/jude/harmony/saved_models/save_pokemon/m0/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m1.keras', '/home/jude/harmony/saved_models/save_pokemon/m1/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m2.keras', '/home/jude/harmony/saved_models/save_pokemon/m2/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m3.keras', '/home/jude/harmony/saved_models/save_pokemon/m3/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m4.keras', '/home/jude/harmony/saved_models/save_pokemon/m4/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m5.keras', '/home/jude/harmony/saved_models/save_pokemon/m5/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m6.keras', '/home/jude/harmony/saved_models/save_pokemon/m6/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m7.keras', '/home/jude/harmony/saved_models/save_pokemon/m7/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m8.keras', '/home/jude/harmony/saved_models/save_pokemon/m8/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m9.keras', '/home/jude/harmony/saved_models/save_pokemon/m9/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m10.keras', '/home/jude/harmony/saved_models/save_pokemon/m10/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m11.keras', '/home/jude/harmony/saved_models/save_pokemon/m11/1/')
# # keras_to_saved_model('/home/jude/harmony/saved_models/pokemon/m12.keras', '/home/jude/harmony/saved_models/save_pokemon/m12/1/')
# 
