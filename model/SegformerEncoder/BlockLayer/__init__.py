import tensorflow as tf
import keras
from keras import ops
import math

from .DropPath import DropPath
from .Attention import MultiHeadAttention
from .FeedForwardNetwork import FeedForwardNetwork

# individual layers of a transformer block
@keras.saving.register_keras_serializable()
class BlockLayer(keras.Model):
    def __init__(
        self,
        dim,
        num_heads,
        sr_ratio,
        attn_drop,
        proj_drop,
        drop_path,
        mlp_ratio,
        **kwargs
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.mlp_ratio = mlp_ratio

        super().__init__(**kwargs)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-05, name = "TBlock_Norm1")
        self.attn = MultiHeadAttention(dim, num_heads, sr_ratio, attn_drop, proj_drop, name = "TBlock_MHA")
        self.drop_path = DropPath(drop_path, name = "TBlock_DropPath")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-05, name = "TBlock_Norm2")
        self.ffn = FeedForwardNetwork(dim, mlp_hidden_dim, drop = proj_drop, name = "TBlock_FFN")

    def compute_output_shape(self, x_shape, **kwargs):
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def build(self, x_shape, W, H):
        self.attn.build(x_shape, H, W)
        self.ffn.build(x_shape, H, W)

        self.expected_output_shape = x_shape
        self.built = True

    def call(self, x, **kwargs):
        attn_output_norm = self.norm1(x)
        attn_output = self.attn(attn_output_norm)
        attn_output_with_drop = self.drop_path(attn_output)

        # print(f"Encoder.call() - {ops.shape(x)} vs {ops.shape(attn_output_with_drop)}")
        # assert ops.shape(x) == ops.shape(attn_output_with_drop)

        x = x + attn_output_with_drop

        ffn_output_norm = self.norm2(x)
        ffn_output = self.ffn(ffn_output_norm)
        ffn_output_with_drop = self.drop_path(ffn_output)

        x = x + ffn_output_with_drop

        return x