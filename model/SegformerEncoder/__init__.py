import tensorflow as tf
from tensorflow import keras
from keras import ops
import math

from .TransformerBlock import TransformerBlock
from typing import NamedTuple

class TransformerBlockParams(NamedTuple):
    depth: int      # L
    patch_size: int # K = P + S
    stride: int     # S
    dim: int        # C
    num_heads: int  # N
    sr_ratio: int   # R - reductio ratio
    mlp_ratio: int  # E
