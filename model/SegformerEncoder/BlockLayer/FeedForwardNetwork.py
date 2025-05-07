import tensorflow as tf
import keras
from keras import ops
import math

from .DepthwiseConv import DepthwiseConvolution

# used in each Block Layer
@keras.saving.register_keras_serializable()
class FeedForwardNetwork(keras.Model):
    def compute_output_shape(self, x_shape, **kwargs):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def build(self, x_shape, H, W):
        assert x_shape[1] == H * W, "Something is very wrong"

        shape = self.fc1.compute_output_shape(x_shape)
        self.dwconv.build(shape, H, W)

        # keras.utils.plot_model(self, show_shapes = True, expand_nested = True, show_layer_names = True)

        self.built = True

    def __init__(
        self, 
        in_features, 
        hidden_features = None, 
        out_features = None, 
        drop = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features if out_features != None else in_features
        self.hidden_features = hidden_features if hidden_features != None else in_features
    
        self.fc1 = keras.layers.Dense(self.hidden_features, name = "FFN_Hidden")
        self.dwconv = DepthwiseConvolution(self.hidden_features, name = "FFN_DepthwiseConv")
        self.activation = keras.layers.Activation("gelu", name = "FFN_Activation")
        self.fc2 = keras.layers.Dense(self.out_features, name = "FFN_output")
        self.drop = keras.layers.Dropout(drop, name = "FFN_dropout")
    
    def get_config(self):
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            **super().get_config(),
        }

    def call(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
