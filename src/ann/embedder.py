from tensorflow.keras import layers, models, applications, losses, optimizers, metrics
import tensorflow as tf

from utils.file_handler.toml import load_model_config
from utils.product_lines import PRODUCTLINES as PLS

def build_embedder(pl: PLS, emb_dim):
    # get config variables for the images
    config = load_model_config(pl)
    config = config['ann']
    img_height = config['img_height']
    img_width = config['img_width']

    inp = layers.Input(shape=(img_height, img_width, 3))
    x = applications.efficientnet_v2.preprocess_input(inp)
    base = applications.EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(emb_dim, use_bias=False, name='embedding')(x)
    emb = tf.math.l2_normalize(x, axis=-1, name='emb_12')

    return models.Model(inp, emb, name='embedder')

