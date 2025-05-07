import keras
from keras import ops
import tensorflow as tf

@keras.saving.register_keras_serializable()
class ResizeLayer(keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.height, self.width, input_shape[3])

    def call(self, inputs):
        resized = ops.image.resize(inputs, size = (self.height, self.width), interpolation = "bilinear")
        return resized
