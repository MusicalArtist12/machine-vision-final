import keras
from keras import ops
import tensorflow as tf
from .ResizeLayer import ResizeLayer
from .PerLayerMLP import PerLayerMLP

@keras.saving.register_keras_serializable()
class LayerMLP(keras.Model):
    def __init__(self, dim):
        super().__init__(name = "LayerMLP")
        self.decode_dim = dim


    def build(self, input_shapes):
        shape = input_shapes[0]
        self.mlps = [
            PerLayerMLP(self.decode_dim, shape[1], shape[2], name = f"PerLayerMLP_{idx}") for idx, _ in enumerate(input_shapes)
        ]

        for input_shape, mlp in zip(input_shapes, self.mlps):
            mlp.build(input_shape)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], self.decode_dim * 4)

    def call(self, inputs):
        outputs = [mlp(x) for mlp, x in zip(self.mlps, inputs)]
        return ops.concatenate(outputs[::-1], axis = 3)

@keras.saving.register_keras_serializable()
class Predictor(keras.Model):
    def __init__(self, dim, num_classes):
        super().__init__(name = "Predictor")
        self.filters = dim
        self.num_classes = num_classes

    def build(self, x_shape):
        # print(f"Predictor - building for shape {x_shape}")
        self.conv = keras.layers.Conv2D(filters = self.filters, kernel_size = 1, use_bias = False, name = "PredictorConv")
        self.bn = keras.layers.BatchNormalization(epsilon = 1e-5, momentum = 0.9, name = "PredictorBN" )
        self.activate = keras.layers.ReLU(name = "PredictorActivation")

        self.dropout = keras.layers.Dropout(0.1, name = "PredictorDropout")
        self.linear_pred = keras.layers.Conv2D(self.num_classes, kernel_size = 1, name = "PredictorFinalConv")

        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        self.call(x)

    def compute_output_shape(self, x_shape):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x
