import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

import registry
from config import TransformerConfig

class RMSNorm(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.gamma = nnx.Param(jnp.ones(config.hidden_size))  # [hidden_size]
        self.epsilon = 1e-6

    def __call__(self , x):
        # x: [batch , seq_len , hidden_size]
        rms = (self.epsilon + jnp.mean(x**2 , axis=-1 , keepdims=True)) ** 0.5  # [batch , seq_len , 1]
        x_norm = x / rms
        return  x_norm * self.gamma

class SwiGLU(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.beta = nnx.Param(jnp.ones())

        d_ff = int((config.hidden_size * 2) / 3)

        self.w1 = nnx.Linear(config.hidden_size , d_ff , use_bias=False , rngs=rngs)
        self.v = nnx.Linear(config.hidden_size , d_ff , use_bias=False , rngs=rngs)
        self.w2 = nnx.Linear(d_ff , config.hidden_size , rngs=rngs)

    def __call__(self , x):
        # Swish(z) = z * (beta * z)
        gate = self.w1(x)
        gate = gate * nnx.sigmoid(self.beta * gate)

        # xV
        value = self.v(x)

        # Element-wise Mult
        return self.w2(gate * value)


# ================================

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
        self.Wo = nnx.Linear(self.hidden_size , self.output_size , use_bias = False , rngs=rngs)

    def __call__(self , x):
        # x.shape is [batch , seq_len , hidden_size]
        Q , K , V = self.Wq(x) , self.Wk(x) , self.Wv(x)  # each is [b , seq_len , hidden_size]

        # Splitting the heads (MULTI-Head Attention)
        def mha_reshape(tensor):
            return rearrange(tensor , 'b n (h d) -> b h n d' , h=self.n_heads)
        Q , K , V = map(mha_reshape , (Q,K,V))  # each is now [batch , head , seq_len , head_dim]

        # Attention
        scale = self.head_size ** -0.5
        attention_weights = jnp.einsum('b h i d , b h j d -> b h i j' , Q , K) * scale # [batch , head , seq_len , seq_len]

        # Causal masking
        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len , seq_len) , dtype=jnp.bool_))  # Make a lower triangular matrix [seq_len , seq_len]
        causal_mask = causal_mask[jnp.newaxis , jnp.newaxis , : , :]  # [1 , 1 , seq_len , seq_len] (so mask can broadcast against attention_weights)

        attention_weights = jnp.where(causal_mask , attention_weights , float('-inf'))
        attention_weights = jax.nn.softmax(attention_weights , axis=-1)
        attention = jnp.einsum('b h i j , b h j d -> b h i d' , attention_weights , V)  # [batch , head , seq_len , head_dim]

        out = rearrange(attention , 'b h n d -> b n (h d)')
        return self.Wo(out)  # [batch , seq_len , hidden_size]

class MultiLayerPerceptron(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size , config.mlp_hidden_size , use_bias=False , rngs=rngs)
        self.fc2 = nnx.Linear(config.mlp_hidden_size , config.hidden_size , use_bias=False , rngs=rngs)

    def __call__(self , x):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nnx.Module):
    def __init__(self , config: TransformerConfig , rngs: nnx.Rngs):
        attn_cls = registry.resolve('attention' , config.attention)
        ffn_cls  = registry.resolve('ffn' , config.ffn)
        norm_cls = registry.resolve('norm' , config.norm)

        self.mhsa = attn_cls(config , rngs=rngs)
        self.mlp = ffn_cls(config , rngs=rngs)
        self.ln1 = norm_cls(config , rngs=rngs)
        self.ln2 = norm_cls(config , rngs=rngs)

    def __call__(self , x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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
        pos_cls = registry.resolve('positional' , config.positional)
        self.pos = pos_cls(config , rngs=rngs)
        self.layers = nnx.List([
            TransformerLayer(config , rngs=rngs) for _ in range(config.n_layers)
        ])
        norm_cls = registry.resolve('norm' , config.norm)
        self.ln_f = norm_cls(config , rngs=rngs)

    def __call__(self , input_ids):
        # input_ids: [batch , seq_len]
        x = self.wte(input_ids)  # [batch , seq_len , hidden_size]
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)  # [batch , seq_len , hidden_size]
        
        wte_matrix = self.wte.embedding.value  # [vocab_size , hidden_size]
        logits = jnp.einsum('b n d , v d -> b n v' , x , wte_matrix)  # [batch , seq_len , vocab_size]
        return logits