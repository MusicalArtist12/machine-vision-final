import keras
from keras import ops

# https://arxiv.org/abs/2105.15203v3 - Segformer (fairly new)
# https://arxiv.org/pdf/2010.11929 - Transformers!
# https://arxiv.org/pdf/1706.03762 - Attention is all you need
# https://github.com/IMvision12/SegFormer-tf/blob/main/models/segformer.py


from .SegformerEncoder import TransformerBlock
from .SegformerEncoder import TransformerBlockParams
from .SegformerDecoder import LayerMLP
from .SegformerDecoder import Predictor
from .SegformerDecoder import ResizeLayer

def Segformer_Decoder(input_shapes, num_classes, batch_size):
    decode_dim = 256

    mlp = LayerMLP(decode_dim)
    predictor = Predictor(decode_dim, num_classes)
    resize = ResizeLayer(input_shapes[0][1], input_shapes[0][2])

    input_layer = keras.layers.Input(shape=input_shapes, batch_size = batch_size)

    x = mlp(input_layer)
    x = predictor(input_layer)
    x = resize(x)

    return keras.Model(inputs = input_layer, outputs = x, name = "Segformer_Decoder")


def Segformer_B0_Encoder(input_shape, num_classes):
    block1 = TransformerBlockParams(2, 7, 4, 32, 1, 8, 8)
    block2 = TransformerBlockParams(2, 3, 2, 64, 2, 4, 8)
    block3 = TransformerBlockParams(2, 3, 2, 160, 5, 2, 4)
    block4 = TransformerBlockParams(2, 3, 2, 256, 8, 1, 4)

    attn_drop = 0.0
    proj_drop = 0.0
    drop_path_rate = 0.0

    block_params = [block1, block2, block3, block4]

    depths = [x.depth for x in block_params]
    dpr = [x for x in ops.linspace(0.0, drop_path_rate, sum(depths))]


    blocks = [
        TransformerBlock(
            param.depth,
            param.patch_size,
            param.stride,
            param.dim,
            param.num_heads,
            param.sr_ratio,
            attn_drop,
            proj_drop,
            dpr,
            sum(depths[0:i]),
            param.mlp_ratio,
            name = f"TransformerBlock_{i}"
        ) for i, param in enumerate(block_params)
    ]

    shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    shapes = []
    for block in blocks:
        block.build(shape)
        # print(f"Segformer - {shape}")
        shape = block.compute_output_shape(shape)
        # print(f"Segformer recieved {shape}")
        shapes.append(shape)


    input_layer = keras.layers.Input(shape=input_shape[1:], batch_size = input_shape[0])

    x = input_layer

    encode_outputs = []

    for block in blocks:
        x = block(x)
        encode_outputs.append(x)

    return keras.Model(inputs = input_layer, outputs = encode_outputs, name = "Segformer_B0_Encoder")
