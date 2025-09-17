import logging

import tensorflow as tf
import keras_cv

from keras_cv import layers as keras_layers

# there is going to be some funky stuff with these imports
from tensorflow.keras import layers, models, Model, Sequential, regularizers, saving, applications

# TODO: rename this file to just model.py


#############
#   PARSE   #
#############
def parse_model_name(model_name: str, height, width, num_classes, weights) -> Model:
    match model_name:
        case 'CnnModelClassic15Mini':
            return CnnModelClassic15Mini(height, width, num_classes)
        case 'CnnModelClassic15':
            return CnnModelClassic15(height, width, num_classes)
        case 'CnnModelClassic15Large': 
            return CnnModelClassic15Large(height, width, num_classes)
        case 'ResNet50V2':
            inputs = layers.Input(shape=(height, width, 3))

            x = PreprocessingLayer(target_size=[height, width])(inputs)
            x = augmentation_pipeline(x)
            
            base = applications.ResNet50V2(
                include_top=False,
                weights=weights,
                input_tensor=x
            )

            x = layers.GlobalAveragePooling2D()(base.output)
            outputs = layers.Dense(num_classes,
                                   activation='softmax',
                                   dtype='float32')(x)

            return Model(inputs, outputs)
        case 'ResNet152':
            inputs = layers.Input(shape=(height, width, 3))

            x = PreprocessingLayer(target_size=[height, width])(inputs)
            x = augmentation_pipeline(x)
            
            base = applications.ResNet152(
                include_top=False,
                weights=weights,
                input_tensor=x
            )

            x = layers.GlobalAveragePooling2D()(base.output)
            outputs = layers.Dense(num_classes,
                                   activation='softmax',
                                   dtype='float32')(x)

            return Model(inputs, outputs)
        case 'EfficientNetV2L':
            inputs = layers.Input(shape=(height, width, 3))

            x = PreprocessingLayer(target_size=[height, width])(inputs)
            x = augmentation_pipeline(x)
            
            base = applications.EfficientNetV2L(
                include_top=False,
                weights=weights,
                input_tensor=x
            )

            x = layers.GlobalAveragePooling2D()(base.output)
            outputs = layers.Dense(num_classes,
                                   activation='softmax',
                                   dtype='float32')(x)

            return Model(inputs, outputs)
        case 'VGG19':
            inputs = layers.Input(shape=(height, width, 3))

            x = PreprocessingLayer(target_size=[height, width])(inputs)
            x = augmentation_pipeline(x)
            
            base = applications.VGG19(
                include_top=False,
                weights=weights,
                input_tensor=x
            )

            x = layers.GlobalAveragePooling2D()(base.output)
            outputs = layers.Dense(num_classes,
                                   activation='softmax',
                                   dtype='float32')(x)

            return Model(inputs, outputs)
        case 'ConvNeXtXLarge':
            inputs = layers.Input(shape=(height, width, 3))

            x = PreprocessingLayer(target_size=[height, width])(inputs)
            x = augmentation_pipeline(x)
            
            base = applications.ConvNeXtXLarge(
                include_top=False,
                weights=weights,
                input_tensor=x
            )

            x = layers.GlobalAveragePooling2D()(base.output)
            outputs = layers.Dense(num_classes,
                                   activation='softmax',
                                   dtype='float32')(x)

            return Model(inputs, outputs)

        case _:
            raise ValueError(f"Unknown model_name: {model_name}")

##############
#   LAYERS   #
##############
@saving.register_keras_serializable(package='cnn')
class PreprocessingLayer(layers.Layer):
    '''
    A non-trainable preprocessing layer that handles image resizing, rescaling, and normalization.
    This layer is active in both training and inference modes.
    '''

    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size

        self.trainable = False
        self.resize_layer = layers.Resizing(*self.target_size)

    def call(self, inputs):
        x = self.resize_layer(inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_size': self.target_size,
            })
        return config
    
    # TODO: I think I can just get rid of this entirely
    @classmethod
    def from_config(cls, config):
        instance = cls(**config)
        return instance

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


##############
#   BLOCKS   #
##############
@saving.register_keras_serializable(package='cnn')
class ConvBnLeakyBlock(layers.Layer):
    '''
    Conv2D + BatchNorm + LeakyReLU + MaxPool, with L2 regularization.
    '''
    def __init__(self, filters, kernel_size, pool_size, l2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size 
        self.l2 = l2

        self.conv = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2))
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(alpha=0.01) 
        self.pool = layers.MaxPooling2D(pool_size)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return self.pool(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'l2': self.l2,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#######################
#   MODEL TEMPLATES   #
#######################
@saving.register_keras_serializable(package='cnn')
class CnnModelClassicBase(Model):
    def __init__(self, height, width, num_classes, **kwargs):
        super().__init__(**kwargs)
        # self.my_input_shape = (1, height, width, 3)
        self.height = height
        self.width = width
        self.num_classes = num_classes 

        self.preprocess = PreprocessingLayer(target_size=[height, width])
        
        self.augment = augmentation_pipeline

        self.blocks = [] 

        self.global_pool = layers.GlobalAveragePooling2D()
        self.hidden = None 
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)

        x = self.augment(x, training=training) # trying new augmentation layer that will work on gpu

        for block in self.blocks:
            x = block(x, training=training)

        x = self.global_pool(x)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def build(self, input_shape):
        logging.warning('input shape: ')
        logging.info(input_shape)

        dummy_input = tf.zeros(input_shape)
        self.call(dummy_input, training=False)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width,
            'num_classes': self.num_classes,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@saving.register_keras_serializable(package='cnn')
class CnnModelClassic15Mini(CnnModelClassicBase):
    def __init__(self, height, width, num_classes, **kwargs):
        super().__init__(height, width, num_classes, **kwargs)

        self.blocks = [
            ConvBnLeakyBlock(filters=16, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=32, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=64, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=128, kernel_size=3, pool_size=2, l2=0.01),
            ]

        self.hidden = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)) 


@saving.register_keras_serializable(package='cnn')
class CnnModelClassic15(CnnModelClassicBase):
    def __init__(self, height, width, num_classes, **kwargs):
        super().__init__(height, width, num_classes, **kwargs)

        self.blocks = [
                ConvBnLeakyBlock(filters=40, kernel_size=3, pool_size=2, l2=0.01),
                ConvBnLeakyBlock(filters=80, kernel_size=3, pool_size=2, l2=0.01),
                ConvBnLeakyBlock(filters=160, kernel_size=3, pool_size=2, l2=0.01),
                ConvBnLeakyBlock(filters=320, kernel_size=3, pool_size=2, l2=0.01),
                ConvBnLeakyBlock(filters=640, kernel_size=3, pool_size=2, l2=0.01),
                ]

        self.hidden = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))


@saving.register_keras_serializable(package='cnn')
class CnnModelClassic15Large(CnnModelClassicBase):
    def __init__(self, height, width, num_classes, **kwargs):
        super().__init__(height, width, num_classes, **kwargs)

        self.blocks = [
            ConvBnLeakyBlock(filters=64, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=128, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=256, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=512, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=768, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=1024, kernel_size=3, pool_size=2, l2=0.01),
            ]

        self.hidden = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))

