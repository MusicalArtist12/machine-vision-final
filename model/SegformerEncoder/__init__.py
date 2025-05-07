import tensorflow as tf
from tensorflow import keras
from keras import ops
import math

from .TransformerBlock import TransformerBlock
from typing import NamedTuple

class TransformerBlockParams(NamedTuple):
    depth: int
    patch_size: int
    stride: int
    dim: int
    num_heads: int
    sr_ratio: int
    mlp_ratio: int
