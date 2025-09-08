import random
import numpy as np
import tensorflow as tf
import requests
from typing import Tuple
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO


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
