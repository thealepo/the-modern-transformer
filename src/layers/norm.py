import jax
import jax.numpy as jnp
from flax import nnx

from config import TransformerConfig


class LayerNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.gamma = nnx.Param(jnp.ones(config.hidden_size))
        self.beta = nnx.Param(jnp.zeros(config.hidden_size))
        self.epsilon = 1e-6

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        mean = jnp.mean(x , axis=-1 , keepdims=True)  # [hidden_size]
        var = jnp.mean((x - mean)**2 , axis=-1 , keepdims=True)  # [hidden_size]

        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)  # [batch , seq_len , hidden_size]
        return x_norm * self.gamma + self.beta

class RMSNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.gamma = nnx.Param(jnp.ones(config.hidden_size))  # [hidden_size]
        self.epsilon = 1e-6

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        rms = (self.epsilon + jnp.mean(x**2 , axis=-1 , keepdims=True)) ** 0.5  # [batch , seq_len , 1]
        x_norm = x / rms
        return x_norm * self.gamma