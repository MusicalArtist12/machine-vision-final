import tensorflow as tf
from typing import NamedTuple
import tensorflow_datasets as tfds
from model import Segformer_B0
import cv2 as cv
import numpy as np
import os
import psutil
import keras
import gc
import cv2 as cv

IMAGE_H = 720 // 4
IMAGE_W = 1280 // 4
NUM_TRANING = 6996
NUM_VAL = 1000
NUM_CLASSES = 1

def normalize_img(image, mask):
    """Normalizes images: `uint8` -> `float32`."""

    # image = [720, 1280, 3], each pixel being [R, G, B], R, G, B in {x in float32 | 0.0 <= x <= 1.0}
    # mask = [720, 1280, 1], each pixel being [P], P in {x in float32 | 0.0 <= x <= 1.0 }
    image = tf.cast(image, tf.float32) / 255.0

    resize = keras.layers.Resizing(720 // 4, 1280 // 4)

    image_resized = resize(image)
    mask_resized = resize(mask)

    return (image_resized, mask_resized)

def load_data(batch_size) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    (train, val) = tfds.load('bdd100k', split = ['train', 'val'], as_supervised = True)

    train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    train = train.batch(batch_size)
    val = val.batch(batch_size)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return train, val
