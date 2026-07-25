import jax
import jax.numpy as jnp
from flax import nnx

from config import TransformerConfig


class LayerNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(config.hidden_size , rngs=rngs)

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        return self.norm(x)

class RMSNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.gamma = nnx.Param(jnp.ones(config.hidden_size))  # [hidden_size]
        self.epsilon = 1e-6

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        rms = (self.epsilon + jnp.mean(x**2 , axis=-1 , keepdims=True)) ** 0.5  # [batch , seq_len , 1]
        x_norm = x / rms
        return x_norm * self.gamma