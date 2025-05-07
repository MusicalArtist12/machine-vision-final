import tensorflow as tf
import keras
from keras import ops
import math

# only used in the FFN
@keras.saving.register_keras_serializable()
class DepthwiseConvolution(keras.Model):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = 3
        self.stride = 1


    def build(self, x_shape, H, W):
        self.reshape_1 = keras.layers.Reshape((H, W, -1), name = "DepthwiseConv_ReshapeIn")
        self.dwconv = keras.layers.Conv2D(
            filters = self.filters,
            kernel_size = self.kernel_size,
            strides = self.stride,
            padding = "same",
            groups = self.filters,
            name = "DepthwiseConv_Conv"
        )
        shape = self.reshape_1.compute_output_shape(x_shape)
        shape = self.dwconv.compute_output_shape(shape)

        self.batch_size = shape[0]
        self.shape_outputted = (shape[1] * shape[2], shape[3])

        self.reshape_2 = keras.layers.Reshape(self.shape_outputted, name = "DepthwiseConv_Reshape_Out")


    def compute_output_shape(self, x_shape, **kwargs):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x, **kwargs):
        x = self.reshape_1(x)
        x = self.dwconv(x)
        shape = ops.shape(x)
        x = self.reshape_2(x)
        return x
