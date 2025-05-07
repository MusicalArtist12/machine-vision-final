import tensorflow as tf
import keras
from keras import ops
import math

# used in the Transformer Block
@keras.saving.register_keras_serializable()
class OverlapPatchEmbeddings(keras.Model):
    def compute_output_shape_with_WH(self, x_shape):
        return self.compute_output_shape(x_shape), self.W, self.H

    def compute_output_shape(self, x_shape):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x):
        # print(f"recieved shape {ops.shape(x)}")
        x = self.conv(self.pad(x))

        x = self.reshape(x)
        x = self.norm(x)
        return x

    def build(self, x_shape):

        # print(f"building patchembedings for {x_shape}")

        # self.batch_size = x_shape[0]
        self.pad = keras.layers.ZeroPadding2D(padding = self.padding, name = "OverlapPatchEmbeddings_Padding")

        self.pad.build(x_shape[1:])

        shape = self.pad.compute_output_shape(x_shape)

        self.conv = keras.layers.Conv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.stride, name = "PatchEmbed_Convolution", trainable = False)

        self.conv.build(shape)
        shape = self.conv.compute_output_shape(shape)

        H = shape[1]
        W = shape[2]
        C = shape[3]

        self.H = H
        self.W = W

        self.reshape = keras.layers.Reshape((H * W, C), name = "PatchEmbed_Reshape")
        self.reshape.build(shape)
        shape = self.reshape.compute_output_shape(shape)

        self.norm = keras.layers.LayerNormalization(epsilon = 1e-05, name = "PatchEmbed_Normalization")
        self.norm.build(shape)

        # this output shape will be (batch_size, H * W, filters)

        self.built = True
        keras.utils.plot_model(self, show_shapes = True, expand_nested = True, show_layer_names = True)


    def __init__(self,
        patch_size,
        stride,
        filters,
        **kwargs
    ):
        super().__init__(**kwargs, trainable = False)
        self.padding = patch_size // 2
        self.kernel_size = patch_size
        self.stride = stride
        self.filters = filters
