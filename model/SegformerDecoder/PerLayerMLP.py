import keras
from keras import ops
import tensorflow as tf

from .ResizeLayer import ResizeLayer

@keras.saving.register_keras_serializable()
class PerLayerMLP(keras.Model):
    def __init__(self, dim, H, W, **kwargs):
        self.W = W
        self.H = H
        self.dim = dim

        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(dim, name = "PerLayerMLP_Dense")

        self.resize_layer = ResizeLayer(H, W, name = "PerLayerMLP_Resize")

    def get_config(self):
        return {
            "dim": self.dim,
            "H": self.H,
            "W": self.W,
            **super().get_config(),
        }

    def build(self, x_shape):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        self.call(x)
        # keras.utils.plot_model(self, show_shapes = True, expand_nested = True, show_layer_names = True)

    def compute_output_shape(self, x_shape):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x):
        x = self.dense(x)
        x = self.resize_layer(x)

        return x
