import jax
import jax.numpy as jnp
from flax import nnx

from config import TransformerConfig


class RotaryPositionEmbedding(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.head_dim = ...
        assert self.head_dim % 2 == 0 , '`self.head_dim` must be divisible by 2'

    
    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        pass