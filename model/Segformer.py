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

def Segformer_B0(input_shape, num_classes):
    block1 = TransformerBlockParams(2, 7, 4, 32, 1, 8, 4)
    block2 = TransformerBlockParams(2, 3, 2, 64, 2, 4, 4)
    block3 = TransformerBlockParams(2, 3, 2, 160, 8, 1, 4)
    block4 = TransformerBlockParams(2, 3, 2, 256, 8, 1, 4)

    attn_drop = 0.0
    proj_drop = 0.0
    drop_path_rate = 0.0

    block_params = [block1, block2, block3, block4]

    depths = [x.depth for x in block_params]
    dpr = [x for x in ops.linspace(0.0, drop_path_rate, sum(depths))]

    
    decode_dim = 256

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

    resize_in = ResizeLayer(input_shape[1] // 2, input_shape[2] // 2)
    shape = (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3])

    shapes = []
    for block in blocks:
        block.build(shape)  
        # print(f"Segformer - {shape}")
        shape = block.compute_output_shape(shape)
        # print(f"Segformer recieved {shape}")
        shapes.append(shape)

    mlp = LayerMLP(decode_dim)
    
    mlp.build(shapes)

    shape = mlp.compute_output_shape(shapes)

    predictor = Predictor(decode_dim, num_classes)
    predictor.build(shape)

    resize = ResizeLayer(720, 1280)
    
    # softmax = keras.layers.Softmax(axis = 2)

    '''
    flatten = keras.layers.Flatten(name = "Decode_Flatten")

    decode_input = keras.layers.Dense(units = 100, name = "Decode_Input")
    decode_hidden = keras.layers.Dense(units = 100, name = "Decode_Hidden")
    decode_output = keras.layers.Dense(units = 40, name = "Decode_Output")
    reshape_final = keras.layers.Reshape((10, 4), name = "Decode_Reshape")
    '''

    input_layer = keras.layers.Input(shape=input_shape[1:], batch_size = input_shape[0])
    x = resize_in(input_layer)

    encode_outputs = []

    for block in blocks:
        x = block(x)
        encode_outputs.append(x)

    x = mlp(encode_outputs)

    x = predictor(x)

    # x = softmax(x)

    '''
    x = flatten(x)
    x = decode_input(x)
    x = decode_hidden(x)
    x = decode_output(x)
    x = reshape_final(x)
    '''

    x = resize(x)

    return keras.Model(inputs = input_layer, outputs = x, name = "Segformer_B0")
'''
class Segformer_B0(keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        block1 = TransformerBlockParams(2, 7, 4, 32, 1, 8, 4)
        block2 = TransformerBlockParams(2, 3, 2, 64, 2, 4, 4)
        block3 = TransformerBlockParams(2, 3, 2, 160, 8, 1, 4)
        block4 = TransformerBlockParams(2, 3, 2, 256, 8, 1, 4)

        attn_drop = 0.0
        proj_drop = 0.0
        drop_path_rate = 0.0

        block_params = [block1, block2, block3, block4]

        depths = [x.depth for x in block_params]
        dpr = [x for x in ops.linspace(0.0, drop_path_rate, sum(depths))]

        decode_dim = 256

        self.blocks = [
            TransformerBlock(
                param.depth,
                param.patch_size,
                param.stride,
                param.dim,Final-Project/test_model.py
                param.num_heads,
                param.sr_ratio,
                attn_drop,
                proj_drop,
                dpr,
                sum(depths[0:i]),
                param.mlp_ratio
            ) for i, param in enumerate(block_params)
        ]

        shape = input_shape
        shapes = []
        for block in self.blocks:
            block.build(shape)  
            # print(f"Segformer - {shape}")
            shape = block.compute_output_shape(shape)
            # print(f"Segformer recieved {shape}")
            shapes.append(shape)

        self.mlp = LayerMLP(decode_dim)
        
        self.mlp.build(shapes)

        shape = self.mlp.compute_output_shape(shapes)

        self.predictor = Predictor(decode_dim, num_classes)
        self.predictor.build(shape)

        self.flatten = keras.layers.Flatten()

        self.decode_dense = keras.layers.Dense(units = 40)
        self.reshape_final = keras.layers.Reshape((10, 4))

        # print("Segformer - finished")
        

    def build(self, input_shape):
        # print(f"Segformer - Building {input_shape}")
        input_layer = keras.layers.Input(shape=input_shape[1:], batch_size = input_shape[0])
        x = self.call(input_layer)

    def call(self, x):
        encode_outputs = []

        for block in self.blocks:
            x = block(x)
            encode_outputs.append(x)

        x = self.mlp(encode_outputs)

        x = self.predictor(x)

        x = self.flatten(x)

        x = self.decode_dense(x)
        x = self.reshape_final(x)

        return x
'''