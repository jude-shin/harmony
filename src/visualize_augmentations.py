import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from keras_cv import layers as keras_layers

IMAGE_PATH = './Selection_001.png'
SAVE_PATH = './augmented.png'
TARGET_SIZE = (413, 313)

img = tf.io.read_file(IMAGE_PATH)
img = tf.image.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, TARGET_SIZE)
img = tf.expand_dims(img, 0)

flip = keras_layers.RandomFlip(
        "horizontal"
        )

# rotate from 180 to 180 
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

# colorjit = keras_layers.RandomColorJitter(
#         value_range=(0, 1),
#         brightness_factor=0.15,
#         contrast_factor=0.20,
#         saturation_factor=0.15,
#         hue_factor=0.05,
#         )

blur = keras_layers.RandomGaussianBlur(
        factor=(1.0, 1.0), 
        kernel_size=15,
        )

# erase = keras_layers.RandomErasing(
#         factor=1.0,
#         scale=(0.02, 0.33),
#         fill_value=None,
#         value_range=(0, 255),
#         seed=None,
#         data_format=None,
#         **kwargs
#         )


pipeline = tf.keras.Sequential([
    keras_layers.RandomApply(contrast, rate=0.7),
    keras_layers.RandomApply(brightness, rate=0.7),
    keras_layers.RandomApply(blur, rate=0.7),
    # keras_layers.RandomApply(erase, rate=0.7),
    keras_layers.RandomApply(upsidedown, rate=0.5),
    shear, rotate, translate, flip, 
    ])

layers_list = [
        ('Original', img),
        ('Rotate', rotate(img, training=True)),
        ('Flip', flip(img, training=True)),
        ('Translate', translate(img, training=True)),
        ('Contrast', contrast(img, training=True)),
        ('Brightness', brightness(img, training=True)),
        ('Blur', blur(img, training=True)),
        ('Shear', shear(img, training=True)),
        # ('Erase', erase(img, training=True)),
        ('Upsidedown', upsidedown(img, training=True)),
        ('Combined', pipeline(img, training=True)),
        ]

cols = 4
rows = (len(layers_list) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()

for ax, (title, img_batch) in zip(axes, layers_list):
    ax.imshow(tf.squeeze(img_batch).numpy().clip(0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

for ax in axes[len(layers_list):]:
    ax.axis('off')

plt.tight_layout()
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
print(f'Saved augmentation grid to {SAVE_PATH}')
plt.show()

