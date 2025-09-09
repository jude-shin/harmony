import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential, layers
from typing import Dict, Text

# TODO: add this to the config file
EMBEDDING_DIMENSION = 32

# ===================================================================

def generate_towers(embedding_dimension: int, query_ds, canidate_ds):
    # movie example: USER (we will eventually QUERY the model with a user)
    # query: user(string)
    # storepass example: IMAGE (we will eventually query the model with an image)
    # query: image(pre-processed vector?)

    # this is a submodel that will be wrapped up in the TwoTowerANN
    # NOTE: figure out what kind of data the preprocessed vector is going to be, and manually generate all the possibilities?
    # ex) 255^(437*313)
    query_tower = Sequential([
        layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
            layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])


    # movie example: MOVIE (we will eventually EXPECT a movie canidate (or a list of the highest ranking movie canidates))
    # canidate: movie(string)
    # storepass example: LABEL (we will eventually EXPECT a label canidate (or a list of the highest ranking label canidates))
    # canidate: label(string)
    # this is a submodel that will be wrapped up in the TwoTowerANN
    canidate_tower = Sequential([
        layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
            layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
            ])

    return query_tower, canidate_tower
    
class TwoTowerANN(tfrs.Model):

  def __init__(self):
    query_tower, canidate_tower = generate_towers(EMBEDDING_DIMENSION)

    super().__init__()


  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

