import tensorflow as tf
import logging

# from tf.keras import layers, models

from tensorflow import keras
from keras import layers, models, Model

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
    """
    A data augmentation layer that applies a random combination of image augmentations.
    This layer is only active during training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1)
                ]
        # TODO: add more augmentation layers if you want 
        self.custom_augmentations = []

    def call(self, inputs, training=False):
        if not training:
            return inputs

        x = inputs
        for layer in self.augmentations + self.custom_augmentations:
            x = layer(x)
        return x

    def add_custom_augmentation(self, layer):
        """
        Add a custom Keras layer to be included in the augmentation pipeline.
        I don't know how useful this may be... we might still want to do some preprocessing in the tf.data.Dataset
        """
        self.custom_augmentations.append(layer)


# BLOCKS
class ConvBlock1(layers.Layer):
    """
    A modular convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool
    """
    def __init__(self, filters, kernel_size=3, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU() # could be LeakyReLu
        self.pool = tf.keras.layers.MaxPooling2D(pool_size)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return self.pool(x)


class FlattenBlock1(layers.Layer):
    """
    Final classifier block: Flatten -> Dense -> optional Dropout -> Output
    """
    def __init__(self, num_classes, hidden_units, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation="relu")
        # add a LeakyReLu?
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)



# MODELS
class ConvModel(Model):
    """
    Full model: Preprocessing -> Augmentation -> ConvBlocks -> FlattenBlock
    """
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = PreprocessingLayer(target_size=input_shape[:2])
        self.augment = AugmentationLayer()

        self.blocks = [
                ConvBlock1(32),
                ConvBlock1(64),
                ConvBlock1(128)
                ]

        self.classifier = FlattenBlock1(num_classes)

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)
        x = self.augment(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        return self.classifier(x, training=training)




