from tensorflow.keras import layers, models, applications, losses, optimizers, metrics, Sequential
import tensorflow as tf

from utils.file_handler.toml import load_model_config
from utils.product_lines import PRODUCTLINES as PLS
from keras_cv import layers as keras_layers

# the embedder turns the raw images into a vector that has seperation. instead of just having each pixel be a dimension of a vector, the embedder helps seperate the vectors that will be in the index

def build_embedder(pl: PLS, emb_dim):
    # get config variables for the images
    config = load_model_config(pl)
    config = config['m0']
    img_height = config['img_height']
    img_width = config['img_width']

    inp = layers.Input(shape=(img_height, img_width, 3))
    x = augmentation_pipeline(inp) 
    x = applications.efficientnet_v2.preprocess_input(inp)
    base = applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(emb_dim, use_bias=False, name='embedding')(x)
    emb = tf.math.l2_normalize(x, axis=-1, name='emb_12')

    return models.Model(inp, emb, name='embedder')


####################
#   AUGMENTAITON   #
####################

flip = keras_layers.RandomFlip(
        "horizontal"
        )

upsidedown = keras_layers.RandomRotation(
        factor=(0.5, 0.5), 
        fill_mode='constant',
        fill_value=0,
        )

rotate = keras_layers.RandomRotation(
        factor=(-0.01, 0.01), 
        fill_mode='constant',
        fill_value=0,
        )

translate = keras_layers.RandomTranslation(
        height_factor=(-0.07, 0.07),
        width_factor=(-0.07, 0.07),
        fill_mode='constant',
        fill_value=0,
        )

shear = keras_layers.RandomShear(
        x_factor=0.10, 
        y_factor=0.05,
        fill_mode='constant',
        fill_value=0,
        )

contrast = keras_layers.RandomContrast(
        value_range=(0, 1), 
        factor=0.50,
        )

brightness = keras_layers.RandomBrightness(
        value_range=(0, 1), 
        factor=0.10,
        )

blur = keras_layers.RandomGaussianBlur(
        factor=(1.0, 1.0), 
        kernel_size=15,
        )

augmentation_pipeline = Sequential([
    keras_layers.RandomApply(contrast, rate=0.7),
    keras_layers.RandomApply(brightness, rate=0.7),
    keras_layers.RandomApply(blur, rate=0.7),
    keras_layers.RandomApply(upsidedown, rate=0.5),
    shear, rotate, translate, flip, 
    ])
