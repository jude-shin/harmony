import tensorflow as tf
import logging

# from tf.keras import layers, models

from tensorflow import keras
from keras import layers, models

# make a parent structure that is a subclass of tf.models


# LAYERS
class PreprocessingLayer(layers.Layer):
    """
    A non-trainable preprocessing layer that handles image resizing, rescaling, and normalization.
    This layer is active in both training and inference modes.
    """

    def __init__(self, target_size=(224, 224), rescale=1./255, mean=0.0, std=1.0, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.target_size = target_size
        self.rescale = rescale
        self.mean = mean
        self.std = std

        self.resize_layer = layers.Resizing(*self.target_size)
        self.rescale_layer = layers.Rescaling(self.rescale)
        self.normalize_layer = layers.Normalization(mean=self.mean, variance=self.std**2)

    def call(self, inputs):
        x = self.resize_layer(inputs)
        x = self.rescale_layer(x)
        x = self.normalize_layer(x)
        return x


class AugmentLayer(layers.Layer):

# BLOCKS
# convolutional block

# MODELS

# make a subclass of the parent class (there will be many of these because these are the different versions)

# 
