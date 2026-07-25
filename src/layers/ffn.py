import jax
import jax.numpy as jnp
from flax import nnx

from config import TransformerConfig


class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size , config.mlp_hidden_size , use_bias=False , rngs=rngs)
        self.fc2 = nnx.Linear(config.mlp_hidden_size , config.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        return x

class SwiGLU(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.beta = nnx.Params(jnp.ones(()))

        d_ff = ((config.mlp_hidden_size * 2) / 3)

        self.w1 = nnx.Linear(config.hidden_size , d_ff , use_bias=False , rngs=rngs)
        self.v = nnx.Linear(config.hidden_size , d_ff , use_bias=False , rngs=rngs)
        self.w2 = nnx.Linear(d_ff , config.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        # Swish(z) = z * sigmoid(beta * z)
        gate = self.w1(x)
        gate = gate * nnx.sigmoid(self.beta * gate)

        # xV
        value = self.v(x)

        # Element-wise multiplication
        return self.w2(gate * value)
        