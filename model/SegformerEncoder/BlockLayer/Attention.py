import tensorflow as tf
import keras
from keras import ops
import math

# https://arxiv.org/pdf/1706.03762 - Attention is all you need
# https://arxiv.org/pdf/2010.11929 - AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE\

# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py 
# lines 96-117

# used in each block 
@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.Model):
    def __init__(self, dim, num_heads, sr_ratio, attn_drop = 0.0, proj_drop = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim             # dimensionality aka the number of layers in the output
        self.num_heads = num_heads # number of heads (h)
        self.head_dim = self.dim // self.num_heads
        self.sr_ratio = sr_ratio   # do we reduce the size of the K V

        self.units = self.num_heads * self.head_dim   # v, k, q units 
        self.sqrt_of_units = math.sqrt(self.head_dim)
        self.attn_drop_val = attn_drop
        self.proj_drop_val = proj_drop

        self.softmax = keras.layers.Softmax(axis = -1, name = "MHA_Softmax")
        self.attn_drop_layer = keras.layers.Dropout(self.attn_drop_val, name = "MHA_Dropout")

        self.proj = keras.layers.Dense(self.dim, name = "MHA_Proj")
        self.proj_drop_layer = keras.layers.Dropout(self.proj_drop_val, name = "MHA_Proj_Drop")

        self.final_reshape = keras.layers.Reshape((-1, self.units), name = "MHA_Reshape_Out")

    def get_config(self):
        return {
            "dim": self.dim,
            "num_heads": self.num_heads,
            "sr_ratio": self.sr_ratio,
            "attn_drop": self.attn_drop_val,
            "proj_drop": self.proj_drop_val,
            **super().get_config(),
        }
        
    def build(self, x_shape, H, W):
        B = x_shape[0]
        N = x_shape[1]
        C = x_shape[2]

        assert x_shape[1] == H * W, "Something is very wrong"

        self.q = keras.Sequential([
            keras.layers.Dense(self.units, name = "Q_Dense"),
            keras.layers.Reshape((N, self.num_heads, self.head_dim), name = "Q_Reshape"),
            keras.layers.Permute((2, 1, 3), name = "Q_Perm")
        ], name = "MHA_Q")

        self.q.build(x_shape)

        if self.sr_ratio > 1: 
            self.sr = keras.Sequential([
                keras.layers.Permute((2, 1), name = "SR_PermIn"),
                keras.layers.Reshape((C, H, W), name = "SR_ReshapeIn"),
                keras.layers.Conv2D(filters = self.dim, kernel_size = self.sr_ratio, strides = self.sr_ratio, name = "SR_Conv"),
                keras.layers.Reshape((C, -1), name = "SR_ReshapeOut"),
                keras.layers.LayerNormalization(epsilon=1e-05, name = "SR_Norm")
            ], name = "MHA_SR")
            self.sr.build(x_shape)

            x_shape = self.sr.compute_output_shape(x_shape)

        self.kv = keras.Sequential([
            keras.layers.Dense(self.units * 2, name = "KV_Dense"),
            keras.layers.Reshape((-1, 2, self.num_heads, self.head_dim), name = "KV_Reshape")
        ], name = "MHA_KV")

        self.kv.build(x_shape)

        self.expected_output_shape = x_shape

        self.built = True
        super().build(x_shape)
        # tf.keras.utils.plot_model(self, show_shapes = True, expand_nested = True, show_layer_names = True)

    def compute_output_shape(self, x_shape, **kwargs):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x):
        get_shape = ops.shape(x)

        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = self.sr(x)

            kv = self.kv(x_)
        else:
            kv = self.kv(x)


        kv = ops.transpose(kv, axes = (2, 0, 3, 1, 4))
        k = ops.transpose(kv[0], axes = (0, 1, 3, 2))
        v = kv[1]

        attn = q @ k
        scale = ops.cast(self.sqrt_of_units, dtype = attn.dtype)
        attn = ops.divide(attn, scale)

        attn = self.softmax(attn)
        attn = self.attn_drop_layer(attn)

        x = attn @ v
        x = ops.transpose(x, axes = (0, 1, 3, 2))

        x = self.final_reshape(x)
        x = self.proj(x)
        x = self.proj_drop_layer(x)

        return x