import random
import numpy as np
import tensorflow as tf
import requests
from typing import Tuple
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO

# TODO: get rid of this stuff lol
def zoom_rotate_img(image):
    """Help: Randomly rotate and zoom the given PIL image degrees and return it"""

    # store initial image size
    initial_size = image.size
    # determine at random how much or little we scale the image
    scale = 0.95 + random.random() * 0.1
    scaled_img_size = tuple([int(i * scale) for i in initial_size])

    # create a blank background with a random color and same size as intial image
    bg_color = tuple(np.random.choice(range(256), size=3))
    background = Image.new("RGB", initial_size, bg_color)

    # determine the center location to place our rotated card
    center_box = tuple((n - o) // 2 for n,
                       o in zip(initial_size, scaled_img_size))

    # scale the image
    scaled_img = image.resize(scaled_img_size)

    # randomly select an angle to skew the image
    max_angle = 5
    skew_angle = random.randint(-max_angle, max_angle)

    # add the scaled image to our color background
    background.paste(
        scaled_img.rotate(skew_angle, fillcolor=bg_color, expand=1).resize(
            scaled_img_size
        ),
        center_box,
    )

    return background


def blur_img(image):
    """Help: Blur the given PIL image and return it"""
    return image.filter(filter=ImageFilter.BLUR)


def adjust_color(image):
    """Help: Randomly reduce or increase the saturation of the provided image and return it"""
    converter = ImageEnhance.Color(image)
    # randomly decide to half or double the image saturation
    saturation = random.choice([0.5, 1.5])
    return converter.enhance(saturation)


def adjust_contrast(image):
    """Help: Randomly decrease or increase the contrast of the provided image and return it"""
    converter = ImageEnhance.Contrast(image)
    # randomly decide to half or double the image saturation
    contrast = random.choice([0.5, 1.5])
    return converter.enhance(contrast)


def adjust_sharpness(image):
    """Help: Randomly decrease or increase the sharpness of the provided image and return it"""
    converter = ImageEnhance.Sharpness(image)
    # randomly decide to half or double the image saturation
    sharpness = random.choice([0.5, 1.5])
    return converter.enhance(sharpness)


# =====================================================


def random_edit_img_old(
    image: Image.Image, distort: bool = True, verbose: bool = False
) -> Image.Image:
    """Help: Make poor edits to the image at random and return the finished copy. Can optionally not distort
    the image if need be."""

    # convert image to RGB if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    if distort:
        # randomly choose which editing operations to perform
        edit_permission = np.random.choice(a=[False, True], size=(4))

        # always skew the image, randomly make the other edits
        image = zoom_rotate_img(image)
        if verbose:
            print("Image skewed")
        if edit_permission[0]:
            image = blur_img(image)
            if verbose:
                print("Image blurred")
        if edit_permission[1]:
            image = adjust_color(image)
            if verbose:
                print("Image color adjusted")
        if edit_permission[2]:
            image = adjust_contrast(image)
            if verbose:
                print("Image contrast adjusted")
        if edit_permission[3]:
            image = adjust_sharpness(image)
            if verbose:
                print("Image sharpness adjusted")

    return image


# =====================================================


def random_edit_img(
    image: Image.Image, distort: bool = True, verbose: bool = False
) -> Image.Image:
    """Help: Make poor edits to the image at random and return the finished copy. Can optionally not distort
    the image if need be, using TensorFlow for GPU acceleration where possible."""

    # always skew the image
    image = zoom_rotate_img(image)
    if verbose:
        print("Image skewed (Note: Skew operation needs custom implementation)")

    # Convert image to TensorFlow tensor
    tensor = get_tensor_from_image(
        image=image, img_width=image.width, img_height=image.height
    )

    if distort:
        edit_permission = np.random.choice(a=[False, True], size=(6))

        if edit_permission[0]:
            tensor = tf.image.random_brightness(tensor, max_delta=0.1)
            if verbose:
                print("Image brightness adjusted")
        if edit_permission[1]:
            tensor = tf.image.random_contrast(tensor, lower=0.8, upper=1.2)
            if verbose:
                print("Image contrast adjusted")
        if edit_permission[2]:
            tensor = tf.image.random_saturation(tensor, lower=0.8, upper=1.2)
            if verbose:
                print("Image color saturation adjusted")
        if edit_permission[3]:
            tensor = tf.image.random_hue(tensor, max_delta=0.04)
            if verbose:
                print("Image hue adjusted")
        if edit_permission[4]:
            gamma = np.random.uniform(0.8, 1.2)
            tensor = tf.image.adjust_gamma(tensor, gamma)
            if verbose:
                print(f"Image gamma adjusted with gamma={gamma}")
        if edit_permission[5]:
            quality = np.random.randint(70, 101)
            image_bytes = tf.image.encode_jpeg(
                tf.cast(tensor * 255, tf.uint8), quality=quality
            )
            tensor = tf.cast(tf.image.decode_image(
                image_bytes), tf.float32) / 255.0
            if verbose:
                print(f"JPEG quality adjusted with quality={quality}")

    tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)
    tensor_uint8 = tf.cast(tensor * 255, tf.uint8)
    image_array = tensor_uint8.numpy()
    image = Image.fromarray(image_array)
    return image


# =====================================================


def preprocess_tensor(image: tf.Tensor, img_width: int, img_height: int) -> tf.Tensor:
    img = tf.image.resize(image, [img_height, img_width])
    img = img / 255.0
    return img


def get_tensor_from_dir(image_path: str, img_width: int, img_height: int) -> tf.Tensor:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = preprocess_tensor(
        image=img, img_width=img_width, img_height=img_height)
    return img


def get_tensor_from_image(
    image: Image.Image, img_width: int, img_height: int
) -> tf.Tensor:
    image_array = np.array(image)
    img = tf.convert_to_tensor(image_array, dtype=tf.float32)
    img = preprocess_tensor(
        image=img, img_width=img_width, img_height=img_height)
    return img


def get_image_from_uri(image_uri: str) -> Image.Image:
    response = requests.get(image_uri)
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    return image
