import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential, layers, Input, Model
from typing import Dict, Text

# TODO: add this to the config file
EMBEDDING_DIMENSION = 32

# ===================================================================

# TODO: the image_encoder is the preprocessing?
def build_query_model(image_encoder, img_height, img_width):
    image_bytes = Input(shape=(), dtype=tf.string, name='image_bytes')
    
    def _decode_and_preprocess(b):
        x = tf.io.decode_image(b, channels=3, expand_animations=False)
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.resize(x, (img_height, img_width), antialias=True)
        return x

    x = tf.map_fn(_decode_and_preprocess, image_bytes, fn_output_signature=tf.float32)
    feats = image_encoder(x)
    feats = tf.math.l2_normalize(feats, axis=-1)
    return Model(image_bytes, feats, name='image_query_model')

def build_scann_index(query_model,
                      label_ids,
                      label_vecs,
                      k,
                      num_leaves,
                      num_leaves_to_search,
                      num_recording_canidates,
                      dimensions_per_block,
                      ):
    index = tfrs.layers.factorized_top_k.ScaNN(
            query_model=query_model,
            k=k,
            distance_measure='dot_product',
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            num_recording_canidates=num_recording_canidates,
            dimensions_per_block=dimensions_per_block,
            )

    ids_tf = tf.constant(label_ids)
    vecs_tf = tf.constant(label_vecs, tf.float32)
    index.index(canidates=vecs_tf, identifiers=ids_tf)

    return index
    
class RetrievalModule(tf.Module):
    def __init__(self, index: tfrs.layers.factorized_top_k.ScaNN):
        super().__init__()
        self.index = index

    @tf.function(
            input_signature=[tf.TensorSpec([None], tf.string, name='image_bytes')]
    )
    def serve(self, image_bytes):
        scores, ids = self.index(image_bytes)
        return {'scores': scores, 'ids': ids}

    @tf.function(
            input_signature=[
                tf.TensorSpec([None], tf.string, name='image_bytes'),
                tf.TensorSpec([], tf.int32, name='k')
            ]
    )
    def serve_with_k(self, image_bytes, k):
        scores, ids = self.index(image_bytes, k=tf.cast(k, tf.int32))
        return {'scores': scores, 'ids': ids}



