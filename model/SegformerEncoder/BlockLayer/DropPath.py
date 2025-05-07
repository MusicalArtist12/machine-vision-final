import tensorflow as tf
import keras
from keras import ops
import math

# used in BlockLayer
@keras.saving.register_keras_serializable()
class DropPath(keras.layers.Layer):

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path


    def call(self, x, training = None):
        if training:
            keep_prop = 1 - self.drop_path
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prop + keras.random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)

            return (x / keep_prop) * random_tensor

        return x
