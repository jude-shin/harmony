import logging

import tensorflow as tf

# there is going to be some funky stuff with these imports
from tensorflow.keras import layers, models, Model, Sequential, regularizers, saving

#############
#   PARSE   #
#############
def parse_model_name(model_name: str, input_shape, num_classes) -> Model:
    match model_name:
        case 'CnnModelClassic15Mini':
            return CnnModelClassic15Mini(input_shape, num_classes)
        case 'CnnModelClassic15':
            return CnnModelClassic15(input_shape, num_classes)
        case 'CnnModelClassic15Large':
            return CnnModelClassic15Large(input_shape, num_classes)

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


##############
#   BLOCKS   #
##############
@saving.register_keras_serializable(package='cnn')
class ConvBlock(layers.Layer):
    '''
    A modular convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool
    '''
    def __init__(self, filters, kernel_size, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.conv = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU() # could be LeakyReLu
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
            'conv': self.conv,
            'bn': self.bn,
            'act': self.act,
            'pool': self.pool,
            })
        return config

    @classmethod
    def from_config(cls, config):
        conv = layers.deserialize(config.pop('conv'))
        bn = layers.deserialize(config.pop('bn'))
        act = layers.deserialize(config.pop('act'))
        pool = layers.deserialize(config.pop('pool'))
    
        instance = cls(**config)
        instance.conv = conv
        instance.bn = bn
        instance.act = act
        instance.pool = pool
        return instance


@saving.register_keras_serializable(package='cnn')
class SEBlock(layers.Layer):
    def __init__(self, ratio, **kw):
        super().__init__(**kw)
        self.ratio = ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = None
        self.dense2 = None
        self.reshape = None

    def build(self, input_shape):
        ch = input_shape[-1]
        self.dense1 = layers.Dense(ch // self.ratio, activation="relu")
        self.dense2 = layers.Dense(ch, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, ch))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@saving.register_keras_serializable(package='cnn')
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'conv1': self.conv1,
            'bn1': self.bn1,
            'relu': self.relu,
            'conv2': self.conv2,
            'bn2': self.bn2,
            })
        return config

    @classmethod
    def from_config(cls, config):
        conv1 = layers.deserialize(config.pop('conv1'))
        bn1 = layers.deserialize(config.pop('bn1'))
        relu = layers.deserialize(config.pop('relu'))
        conv2 = layers.deserialize(config.pop('conv2'))
        bn2 = layers.deserialize(config.pop('bn2'))
    
        instance = cls(**config)
        instance.conv1 = conv1
        instance.bn1 = bn1
        instance.relu = relu
        instance.conv2 = conv2
        instance.bn2 = bn2
        return instance


@saving.register_keras_serializable(package='cnn')
class DropBlock(layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

        self.drop = layers.SpatialDropout2D(rate)

    def call(self, inputs, training=False):
        return self.drop(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'drop': self.drop,
            })
        return config

    @classmethod
    def from_config(cls, config):
        drop = layers.deserialize(config.pop('drop'))
        instance = cls(**config)
        instance.drop = drop
        return instance


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
        self.act = layers.LeakyReLU(negative_slope=0.01) # TODO: add this?
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
            'conv': self.conv,
            'bn': self.bn,
            'act': self.act,
            'pool': self.pool,
            })
        return config

    @classmethod
    def from_config(cls, config):
        conv = layers.deserialize(config.pop('conv'))
        bn = layers.deserialize(config.pop('bn'))
        act = layers.deserialize(config.pop('act'))
        pool = layers.deserialize(config.pop('pool'))
    
        instance = cls(**config)
        instance.conv = conv
        instance.bn = bn
        instance.act = act
        instance.pool = pool
        return instance

# @saving.register_keras_serializable(package='cnn')
# class DenseDropoutBlock(layers.Layer):
#     '''
#     Dense + Dropout with L2 regularization, for classifier head.
#     '''
#     def __init__(self, units, dropout_rate=0.5, l2=0.01, **kwargs):
#         super().__init__(**kwargs)
#         self.dense = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2))
#         self.dropout = layers.Dropout(dropout_rate)
# 
#     def call(self, inputs, training=False):
#         x = self.dense(inputs)
#         return self.dropout(x, training=training)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'dense': self.dense,
#             'dropout': self.dropout,
#             })
#         return config
# 
#     @classmethod
#     def from_config(cls, config):
#         dense = layers.deserialize(config.pop('dense'))
#         dropout = layers.deserialize(config.pop('dropout'))
#     
#         instance = cls(**config)
#         instance.dense = dense
#         instance.dropout = dropout
#         return instance

####################
#   BLOCK MACROS   #
####################

def makeCnnSeq(input_shape, filters, dropout_rate):
    return Sequential([
        ConvBlock(filters=filters, kernel_size=3, pool_size=2),
        ResidualBlock(filters, kernel_size=3),
        SEBlock(8),
        DropBlock(rate=dropout_rate)
        ])

#######################
#   MODEL TEMPLATES   #
#######################
# @saving.register_keras_serializable()
# class CnnModel1(Model):
#     def __init__(self, input_shape, num_classes, **kwargs):
#         super().__init__(**kwargs)
#         self.input_shape = input_shape
#         self.num_classes = num_classes 
# 
#         self.preprocess = PreprocessingLayer(target_size=input_shape[1:3])
# 
#         self.blocks = [
#                 makeCnnSeq(input_shape, 32, 0.2),
#                 makeCnnSeq(input_shape, 64, 0.3),
#                 makeCnnSeq(input_shape, 128, 0.4),
#                 ]
# 
#         self.pool = layers.GlobalAveragePooling2D()
#         self.output_layer = layers.Dense(num_classes, activation='softmax')
# 
#     def call(self, inputs, training=False):
#         x = self.preprocess(inputs)
# 
#         for block in self.blocks: 
#             x = block(x, training=training)
# 
#         x = self.pool(x)
#         return self.output_layer(x)
# 
#     def build(self, input_shape):
#         logging.info('INPUT SHAPE')
#         logging.info(input_shape)
#         dummy_input = tf.zeros(input_shape)
#         self.call(dummy_input, training=False)
#         super().build(input_shape)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'input_shape': self.input_shape,
#             'preprocess': self.preprocess,
#             'blocks': self.blocks,
#             'pool': self.pool,
#             'output_layer': self.output_layer,
#             })
#         return config
# 
#     @classmethod
#     def from_config(cls, config):
#         preprocess = layers.deserialize(config.pop('preprocess'))
#         blocks = [layers.deserialize(block_cfg) for block_cfg in config.pop('blocks')]
#         pool = layers.deserialize(config.pop('pool'))
#         output_layer = layers.deserialize(config.pop('output_layer'))
#         input_shape = [1, *config.pop('input_shape')]
#     
#         instance = cls(input_shape=input_shape, num_classes=output_layer.units, **config)
#         instance.preprocess = preprocess
#         instance.blocks = blocks
#         instance.pool = pool
#         instance.output_layer = output_layer
# 
#         instance.build(input_shape)
#         return instance


@saving.register_keras_serializable()
class CnnModelClassic15Mini(Model):
    '''
    Smaller version of CnnModelClassic15 to reduce overfitting:
    - Fewer filters per ConvBnLeakyBlock
    - One less block
    - Smaller Dense layer in head
    '''
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes 

        self.preprocess = PreprocessingLayer(target_size=input_shape[1:3])

        self.blocks = [
            ConvBnLeakyBlock(filters=16, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=32, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=64, kernel_size=3, pool_size=2, l2=0.01),
            ConvBnLeakyBlock(filters=128, kernel_size=3, pool_size=2, l2=0.01),
            ]

        self.global_pool = layers.GlobalAveragePooling2D()
        self.hidden = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self.dropout = layers.Dropout(0.5)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.preprocess(inputs)

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
            'input_shape': self.input_shape,
            'preprocess': self.preprocess,
            'blocks': self.blocks,
            'global_pool': self.global_pool,
            'hidden': self.hidden,
            'dropout': self.dropout,
            'output_layer': self.output_layer,
            })
        return config

    @classmethod
    def from_config(cls, config):
        preprocess = layers.deserialize(config.pop('preprocess'))
        blocks = [layers.deserialize(cfg) for cfg in config.pop('blocks')]
        global_pool = layers.deserialize(config.pop('global_pool'))
        hidden = layers.deserialize(config.pop('hidden'))
        dropout = layers.deserialize(config.pop('dropout'))
        output_layer = layers.deserialize(config.pop('output_layer'))

        input_shape = config.pop('input_shape')
        input_shape = [1, *input_shape[-3:]]
    
        instance = cls(input_shape=input_shape, num_classes=output_layer.units, **config)
        instance.preprocess = preprocess
        instance.blocks = blocks
        instance.global_pool = global_pool
        instance.hidden = hidden
        instance.dropout = dropout
        instance.output_layer = output_layer

        instance.build(input_shape)
        return instance



# @saving.register_keras_serializable()
# class CnnModelClassic15(Model):
#     '''
#     Improved model based on "model_classic_15":
#     - Uses Conv + BN + LeakyReLU blocks
#     - Includes spatial downsampling
#     - Ends with GlobalAveragePooling + Dense classification head
#     '''
#     def __init__(self, input_shape, num_classes, **kwargs):
#         super().__init__(**kwargs)
#         self.input_shape = input_shape
#         self.num_classes = num_classes
# 
#         self.preprocess = PreprocessingLayer(target_size=input_shape[1:3])
# 
#         self.blocks = [
#                 ConvBnLeakyBlock(filters=40, kernel_size=3, pool_size=2, l2=0.01),
#                 ConvBnLeakyBlock(filters=80, kernel_size=3, pool_size=2, l2=0.01),
#                 ConvBnLeakyBlock(filters=160, kernel_size=3, pool_size=2, l2=0.01),
#                 ConvBnLeakyBlock(filters=320, kernel_size=3, pool_size=2, l2=0.01),
#                 ConvBnLeakyBlock(filters=640, kernel_size=3, pool_size=2, l2=0.01),
#                 ]
# 
#         self.global_pool = layers.GlobalAveragePooling2D()
#         self.hidden = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))
#         self.dropout = layers.Dropout(0.5)
#         self.output_layer = layers.Dense(num_classes, activation='softmax')
# 
#     def call(self, inputs, training=False):
#         x = self.preprocess(inputs)
# 
#         for block in self.blocks:
#             x = block(x, training=training)
# 
#         x = self.global_pool(x)
#         x = self.hidden(x)
#         x = self.dropout(x, training=training)
#         return self.output_layer(x)
# 
#     def build(self, input_shape):
#         dummy_input = tf.zeros(input_shape)
#         self.call(dummy_input, training=False)
#         super().build(input_shape)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'input_shape': self.input_shape,
#             'preprocess': self.preprocess,
#             'blocks': self.blocks,
#             'global_pool': self.global_pool,
#             'hidden': self.hidden,
#             'dropout': self.dropout,
#             'output_layer': self.output_layer,
#             })
#         return config
# 
#     @classmethod
#     def from_config(cls, config):
#         preprocess = layers.deserialize(config.pop('preprocess'))
#         blocks = [layers.deserialize(cfg) for cfg in config.pop('blocks')]
#         global_pool = layers.deserialize(config.pop('global_pool'))
#         hidden = layers.deserialize(config.pop('hidden'))
#         dropout = layers.deserialize(config.pop('dropout'))
#         output_layer = layers.deserialize(config.pop('output_layer'))
# 
#         input_shape = [1, *config.pop('input_shape')]
#     
#         instance = cls(input_shape=input_shape, num_classes=output_layer.units, **config)
#         instance.preprocess = preprocess
#         instance.blocks = blocks
#         instance.global_pool = global_pool
#         instance.hidden = hidden
#         instance.dropout = dropout
#         instance.output_layer = output_layer
# 
#         instance.build(input_shape)
#         return instance
# 
# 
# @saving.register_keras_serializable()
# class CnnModelClassic15Large(Model):
#     def __init__(self, input_shape, num_classes, **kwargs):
#         super().__init__(**kwargs)
#         self.input_shape = input_shape
#         self.num_classes = num_classes
# 
#         self.preprocess = PreprocessingLayer(target_size=input_shape[1:3])
# 
#         self.blocks = [
#             ConvBnLeakyBlock(filters=64, kernel_size=3, pool_size=2, l2=0.01),
#             ConvBnLeakyBlock(filters=128, kernel_size=3, pool_size=2, l2=0.01),
#             ConvBnLeakyBlock(filters=256, kernel_size=3, pool_size=2, l2=0.01),
#             ConvBnLeakyBlock(filters=512, kernel_size=3, pool_size=2, l2=0.01),
#             ConvBnLeakyBlock(filters=768, kernel_size=3, pool_size=2, l2=0.01),
#             ConvBnLeakyBlock(filters=1024, kernel_size=3, pool_size=2, l2=0.01),
#             ]
# 
#         self.global_pool = layers.GlobalAveragePooling2D()
#         self.hidden = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))
#         self.dropout = layers.Dropout(0.5)
#         self.output_layer = layers.Dense(num_classes, activation='softmax')
# 
#     def call(self, inputs, training=False):
#         x = self.preprocess(inputs)
# 
#         for block in self.blocks:
#             x = block(x, training=training)
# 
#         x = self.global_pool(x)
#         x = self.hidden(x)
#         x = self.dropout(x, training=training)
#         return self.output_layer(x)
# 
#     def build(self, input_shape):
#         dummy_input = tf.zeros(input_shape)
#         self.call(dummy_input, training=False)
#         super().build(input_shape)
# 
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'input_shape': self.input_shape,
#             'preprocess': self.preprocess,
#             'blocks': self.blocks,
#             'global_pool': self.global_pool,
#             'hidden': self.hidden,
#             'dropout': self.dropout,
#             'output_layer': self.output_layer,
#             })
#         return config
# 
#     @classmethod
#     def from_config(cls, config):
#         preprocess = layers.deserialize(config.pop('preprocess'))
#         blocks = [layers.deserialize(cfg) for cfg in config.pop('blocks')]
#         global_pool = layers.deserialize(config.pop('global_pool'))
#         hidden = layers.deserialize(config.pop('hidden'))
#         dropout = layers.deserialize(config.pop('dropout'))
#         output_layer = layers.deserialize(config.pop('output_layer'))
#         input_shape = [1, *config.pop('input_shape')]
# 
#         instance = cls(input_shape=input_shape, num_classes=output_layer.units, **config)
#         instance.preprocess = preprocess
#         instance.blocks = blocks
#         instance.global_pool = global_pool
#         instance.hidden = hidden
#         instance.dropout = dropout
#         instance.output_layer = output_layer
# 
#         instance.build(input_shape)
#         return instance
# 
