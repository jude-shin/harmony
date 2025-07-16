import logging

# from tf.keras import layers, models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model, Sequential # there is going to be some funky stuff with these imports


# LAYERS
class PreprocessingLayer(layers.Layer):
    '''
    A non-trainable preprocessing layer that handles image resizing, rescaling, and normalization.
    This layer is active in both training and inference modes.
    '''

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
    '''
    A data augmentation layer that applies a random combination of image augmentations.
    This layer is only active during training.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = [
                layers.RandomFlip('horizontal_and_vertical'),
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
        '''
        Add a custom Keras layer to be included in the augmentation pipeline.
        I don't know how useful this may be... we might still want to do some preprocessing in the tf.data.Dataset
        '''
        self.custom_augmentations.append(layer)


# BLOCKS
class ConvBlock(layers.Layer):
    '''
    A modular convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool
    '''
    def __init__(self, filters, kernel_size=3, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU() # could be LeakyReLu
        self.pool = layers.MaxPooling2D(pool_size)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return self.pool(x)


class SEBlock(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(channel_dim // self.ratio, activation='relu')
        self.dense2 = layers.Dense(channel_dim, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, channel_dim))

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return inputs * x

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return self.relu(x + inputs)

class FlattenBlock(layers.Layer):
    '''
    Final classifier block: Flatten -> Dense -> optional Dropout -> Output
    '''
    def __init__(self, num_classes, hidden_units, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        # add a LeakyReLu?
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

class GlobalPoolBlock(layers.Layer):
    '''
    Classifier alternative to flattenning 
    '''
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.pool(inputs)
        return self.output_layer(x)


def makeCnnSeq(filters, dropout_rate):
    return Sequential([
        ConvBlock(filters),
        ResidualBlock(filters),
        SEBlock(),
        DropBlock(rate=dropout_rate)
        ])

class DropBlock(layers.Layer):
    def __init__(self, rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.drop = layers.SpatialDropout2D(rate)

    def call(self, inputs, training=False):
        return self.drop(inputs, training=training)



# MODELS
class CnnModel1(Model):
    '''
    Full model: Preprocessing -> Augmentation -> ConvBlocks -> FlattenBlock
    '''
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.preprocess = PreprocessingLayer(target_size=input_shape[:2])
        self.augment = AugmentLayer()

        self.blocks = [
                makeCnnSeq(32, 0.2),
                makeCnnSeq(64, 0.3),
                makeCnnSeq(128, 0.4),
                ]

        self.pool = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)
        x = self.augment(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.pool(x)
        return self.output_layer(x)


