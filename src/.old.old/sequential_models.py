


from tensorflow.keras import initializers, layers, models, regularizers# type: ignore

# CLASSIC MODELS
def model_classic_1(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(512, (3, 3)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(2048))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_2(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_3(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.ELU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_4(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_5(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(128, (3, 3)))  # Increased filters from 64 to 128
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(256, (3, 3)))  # Increased filters from 128 to 256
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(512, (3, 3)))  # Increased filters from 256 to 512
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3)))  # Increased filters from 256 to 512
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(1024, (3, 3)))  # Increased filters from 512 to 1024
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))  # Increased units from 2048 to 4096
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_6(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(96, (3, 3)))  # Increased filters from 64 to 96
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(192, (3, 3)))  # Increased filters from 128 to 192
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(384, (3, 3)))  # Increased filters from 256 to 384
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(384, (3, 3)))  # Increased filters from 256 to 384
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(768, (3, 3)))  # Increased filters from 512 to 768
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(3072))  # Increased units from 2048 to 3072
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_8(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, kernel_regularizer=regularizers.l2(0.001)))  # Added L2 regularization
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))  # Increased dropout rate from 0.5 to 0.6
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_9(img_width, img_height, unique_classes):
    # reduceing the dropout while testing
    # also there is one more layer to this model
    # not just larger neurons
    # this used Glorog Init
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Second Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Third Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fourth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Fifth Convolutional Block
    model.add(layers.Conv2D(512, (3, 3), kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D(2, 2))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))  # Increased units
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001), kernel_initializer=initializers.glorot_uniform()))  # Added another dense layer
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform()))

    return model

def model_classic_10(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers with Dropout and L2 Regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_11(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers with Dropout and L2 Regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_12(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Added Fifth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers with Dropout and L2 Regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_13(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fifth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Sixth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(1024, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_14(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(36, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 32 to 36
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(72, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 64 to 72
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(144, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 128 to 144
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(288, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 256 to 288
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fifth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(576, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 512 to 576
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Fully Connected Layers with Dropout and L2 Regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(144, kernel_regularizer=regularizers.l2(0.01)))  # Increased from 128 to 144
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_classic_15(img_width, img_height, unique_classes):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # First Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(40, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 36 to 40
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(80, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 72 to 80
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(160, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 144 to 160
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fourth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(320, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 288 to 320
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Fifth Convolutional Block with Batch Normalization and L2 Regularization
    model.add(layers.Conv2D(640, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 576 to 640
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(640, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # Increased from 576 to 640
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model





# RESNET MODELS
def model_9(img_width, img_height, unique_classes):
    # resnet-like model
    # did not work for pokemon (lr 0.0001)
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust the shortcut to have the same shape as the output
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.001))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.add([shortcut, x])
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        return x

    inputs = layers.Input(shape=(img_height, img_width, 3))  # Corrected input shape
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters, stride=2 if filters != 64 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(unique_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def model_91(img_width, img_height, unique_classes):
    # resnet-like model 
    # uses He initalization
    # did not work for pokemon (lr 0.0001)

    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                          kernel_regularizer=regularizers.l2(0.001),
                          kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', 
                          kernel_regularizer=regularizers.l2(0.001),
                          kernel_initializer=initializers.he_normal())(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust the shortcut to have the same shape as the output
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', 
                                     kernel_regularizer=regularizers.l2(0.001),
                                     kernel_initializer=initializers.he_normal())(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.add([shortcut, x])
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        return x

    inputs = layers.Input(shape=(img_height, img_width, 3))  # Corrected input shape
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', 
                      kernel_regularizer=regularizers.l2(0.001),
                      kernel_initializer=initializers.he_normal())(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters, stride=2 if filters != 64 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer=initializers.he_normal())(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(unique_classes, activation='softmax',
                           kernel_initializer=initializers.he_normal())(x)

    model = models.Model(inputs, outputs)
    return model

def model_10(img_width, img_height, unique_classes):
    # vgg-like model
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(img_width, img_height, 3)))

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.LeakyReLU(negative_slope=0.01))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(unique_classes, activation='softmax'))

    return model

def model_11(img_width, img_height, unique_classes):
    # inception-like model
    def inception_module(x, filters):
        branch1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)

        branch2 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        branch2 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(branch2)

        branch3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        branch3 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(branch3)

        branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch4 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(branch4)

        return layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    inputs = layers.Input(shape=(img_width, img_height, 3))
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = inception_module(x, 64)
    x = inception_module(x, 128)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = inception_module(x, 256)
    x = inception_module(x, 512)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(unique_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model