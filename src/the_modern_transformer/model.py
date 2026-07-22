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
        self.Wq = nnx.Linear(self.hidden_size , self.hidden_size , use_bias = False , rngs=rngs)
        self.Wk = nnx.Linear(self.hidden_size , self.hidden_size , use_bias = False , rngs=rngs)
        self.Wv = nnx.Linear(self.hidden_size , self.hidden_size , use_bias = False , rngs=rngs)
        self.Wo = nnx.Linear(...)

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        Q , K , V = self.Wq(x) , self.Wk(x) , self.Wv(x)  # each is [b , seq_len , hidden_size]

        # we want to split heads
        def mha_reshape(tensor):
            rearrange(tensor , 'b s (h d) -> b h s d' , h=self.n_heads)
        Q , K , V = map(mha_reshape , (Q,K,V))  # each is now [batch , head , seq_len , head_dim]

        # Self-Attention Equ
        Q_K_dotted = jnp.einsum('b h i d , b h j d -> b h i j' , Q , K) # [batch , head , seq_len , seq_len]
        scale = self.head_dim ** -0.5
        inner = Q_K_dotted * scale # [batch , head , seq_len , seq_len]
        softmaxed = jax.nn.softmax(inner , axis=-1)
        attention = jnp.einsum('b h i j , b h j d -> b h i d' , softmaxed , V)

        out = rearrange(attention , 'b h s d -> b s (h d)')
        return self.Wo(out)

class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(config.HIDDEN_SIZE , config.MLP_HIDDEN_SIZE , use_bias=False , rngs=rngs)
        self.layer2 = nnx.Linear(config.MLP_HIDDEN_SIZE , config.HIDDEN_SIZE , use_bias=False , rngs=rngs)

    def __call__(self , x):
        x = self.layer1(x)
        x = nnx.gelu(x)
        x = self.layer2(x)
        return x

class TransformerLayer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.mhsa = MultiHeadAttention(config , rngs=rngs)
        self.mlp = MultiLayerPerceptron(config , rngs=rngs)
        self.ln1 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)
        self.ln2 = nnx.LayerNorm(config.HIDDEN_SIZE , rngs=rngs)

    def __call__(self , x):
        x = x + self.ln1(self.mhsa(x))
        x = x + self.ln2(self.mlp(x))
        return x