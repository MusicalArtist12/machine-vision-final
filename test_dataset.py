# Julia Abdel-Monem
# CS455 Machine Vision Spring 2025
# Requires Python 3.11, Tensorflow

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

class Dataset(NamedTuple):
    train_images: any
    train_labels: any
    test_images: any
    test_labels: any
    class_names: list[str]

def normalize_img(image, objects):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., objects

def load_data() -> Dataset:
    (train, test, val) = tfds.load('bd100k', split = ['train', 'test', 'val'], as_supervised = True)


    train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    
    train = train.cache()
    test = test.cache()
    val = val.cache()

    train = train.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

def main():
    data = load_data()
    
if __name__ == '__main__':
    main()