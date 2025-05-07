import tensorflow as tf
from .OverlapPatchEmbeddings import OverlapPatchEmbeddings
from .BlockLayer import BlockLayer

import keras
from keras import ops

# https://arxiv.org/pdf/2105.15203v3
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.Model):
    def __init__(
        self,
        depth,      # depth here is xN in the Xie, E. et al. page 3
        patch_size,
        stride,
        dim,
        num_heads,
        sr_ratio,
        attn_drop,
        proj_drop,
        dpr,
        current_rate,
        mlp_ratio,
        **kwargs
    ):
        self.depth = depth
        self.patch_size = patch_size
        self.stride = stride
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.dpr = dpr
        self.current_rate = current_rate
        self.mlp_ratio = mlp_ratio

        super().__init__(**kwargs)
        self.patch_embed = OverlapPatchEmbeddings(patch_size, stride, dim)
        self.block_layers = [
            BlockLayer(
                dim,
                num_heads,
                sr_ratio,
                attn_drop,
                proj_drop,
                dpr[current_rate + i],
                mlp_ratio
            ) for i in range(depth)
        ]
        self.norm = keras.layers.LayerNormalization(epsilon=1e-05)


    def build(self, x_shape):
        self.patch_embed.build(x_shape)
        shape, W, H = self.patch_embed.compute_output_shape_with_WH(x_shape)

        for layer in self.block_layers:
            layer.build(shape, W, H)

        self.reshape = keras.layers.Reshape((H, W, -1))
        self.expected_output_shape = self.reshape.compute_output_shape(self.block_layers[0].compute_output_shape(shape))

        self.built = True

        # keras.utils.plot_model(self, show_shapes = True, expand_nested = True, show_layer_names = True)

        # print(f"PatchEncoder.build() - {x_shape} -> {self.expected_output_shape}")

    def compute_output_shape(self, x_shape):
        # print(f"transformer block (compute) - {x_shape}")
        x = keras.layers.Input(x_shape[1:], batch_size = x_shape[0])
        x = self.call(x)
        return ops.shape(x)

    def call(self, x):
        # print(f"transformer block - {ops.shape(x)}")
        x = self.patch_embed(x)
        for i, layer in enumerate(self.block_layers):
            x = layer(x)

        x = self.norm(x)
        x = self.reshape(x)

        return x