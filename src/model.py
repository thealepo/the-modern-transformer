import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

import registry
from config import TransformerConfig


class LayerNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(config.hidden_size , rngs=rngs)

    def __call__(self , x):
        # x: [batch , seq_len , hidden_size]
        return self.norm(x)


class MultiHeadAttention(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // config.n_heads
        self.hidden_size = config.hidden_size
        self.output_size = config.hidden_size

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
        self.layer1 = nnx.Linear(config.hidden_size , config.mlp_hidden_size , use_bias=False , rngs=rngs)
        self.layer2 = nnx.Linear(config.mlp_hidden_size , config.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        x = self.layer1(x)
        x = nnx.gelu(x)
        x = self.layer2(x)
        return x

class TransformerLayer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        # slot selection: config names the variant, registry hands back the class
        attn_cls = registry.resolve("attention" , config.attention)
        ffn_cls  = registry.resolve("ffn" , config.ffn)
        norm_cls = registry.resolve("norm" , config.norm)

        self.mhsa = attn_cls(config , rngs=rngs)
        self.mlp = ffn_cls(config , rngs=rngs)
        self.ln1 = norm_cls(config , rngs=rngs)
        self.ln2 = norm_cls(config , rngs=rngs)

    def __call__(self , x):
        x = x + self.ln1(self.mhsa(x))
        x = x + self.ln2(self.mlp(x))
        return x

class LearnedPositional(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.wpe = nnx.Embed(config.seq_len , config.hidden_size , rngs=rngs)

    def __call__(self , x):
        # x: [batch , seq_len , hidden_size]
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)
        return x + self.wpe(positions)


class Transformer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.wte = nnx.Embed(config.vocab_size , config.hidden_size , rngs=rngs)
        pos_cls = registry.resolve("positional" , config.positional)
        self.pos = pos_cls(config , rngs=rngs)
        self.layers = nnx.List([
            TransformerLayer(config , rngs=rngs) for _ in range(config.n_layers)
        ])

    def __call__(self , input_ids):
        # input_ids: [batch , seq_len]
        x = self.wte(input_ids)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x)

        return x