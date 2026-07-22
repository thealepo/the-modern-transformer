from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

@dataclass(frozen=True , kw_only=True , slots=True)
class TransformerConfig:
    VOCAB_SIZE = 256
    SEQ_LEN = 32
    HIDDEN_SIZE = 64
    MLP_HIDDEN_SIZE = 4 * 64
    N_HEADS = 4
    N_LAYERS = 2

class MultiHeadAttention(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.n_heads = config.N_HEADS
        self.head_size = config.HIDDEN_SIZE // config.N_HEADS
        self.hidden_size = config.HIDDEN_SIZE
        self.output_size = config.HIDDEN_SIZE

        # matrices
        self.Wq = nnx.Linear(...)
        self.Wk = nnx.Linear(...)
        self.Wv = nnx.Linear(...)
        self.Wo = nnx.Linear(...)

    def __call__(self , x):
