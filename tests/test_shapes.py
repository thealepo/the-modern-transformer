import dataclasses

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import registry
from config import TransformerConfig, _VALID_VARIANTS
from model import Transformer

BATCH = 2
ALL_VARIANTS = [(slot , name) for slot , names in registry.VARIANTS.items() for name in names]

def test_registry_and_config_agree_on_variants():
    assert registry.VARIANTS == _VALID_VARIANTS

@pytest.mark.parametrize('slot,name' , ALL_VARIANTS)
def test_registry_resolves_every_declared_variant(slot , name):
    assert registry.resolve(slot , name) is not None

@pytest.mark.parametrize('slot,name' , ALL_VARIANTS)
def test_component_preserves_activation_shape(slot , name , tiny_config):
    config = dataclasses.replace(tiny_config , **{slot: name})
    component = registry.resolve(slot , name)(config , rngs=nnx.Rngs(0))

    x = jnp.ones((BATCH , config.seq_len , config.hidden_size))
    assert component(x).shape == x.shape

@pytest.mark.parametrize('slot,name' , ALL_VARIANTS)
def test_component_jits(slot , name , tiny_config):
    config = dataclasses.replace(tiny_config , **{slot: name})
    component = registry.resolve(slot , name)(config , rngs=nnx.Rngs(0))
    graphdef , state = nnx.split(component)

    x = jnp.ones((BATCH , config.seq_len , config.hidden_size))
    jitted = jax.jit(lambda state , x: nnx.merge(graphdef , state)(x))
    assert jitted(state , x).shape == x.shape

@pytest.mark.parametrize('slot,name' , ALL_VARIANTS)
def test_full_model_with_each_variant(slot , name , tiny_config):
    config = dataclasses.replace(tiny_config , **{slot: name})
    model = Transformer(config , rngs=nnx.Rngs(0))

    tokens = jnp.zeros((BATCH , config.seq_len) , dtype=jnp.int32)
    logits = model(tokens)

    assert logits.shape == (BATCH , config.seq_len , config.vocab_size)
    assert jnp.all(jnp.isfinite(logits))

def test_model_jit_matches_eager(tiny_config):
    model = Transformer(tiny_config , rngs=nnx.Rngs(0))
    graphdef , state = nnx.split(model)
    tokens = jnp.arange(BATCH * tiny_config.seq_len).reshape(BATCH , tiny_config.seq_len)
    tokens = tokens % tiny_config.vocab_size

    jitted = jax.jit(lambda state , tokens: nnx.merge(graphdef , state)(tokens))
    eager , compiled = model(tokens) , jitted(state , tokens)

    drift = jnp.abs(eager - compiled).max() / jnp.std(eager)
    assert drift < 1e-2 , f'jit/eager drift {drift:.2e} of output scale'

def test_attention_is_causal(tiny_config):
    model = Transformer(tiny_config , rngs=nnx.Rngs(0))
    tokens = jnp.ones((1 , tiny_config.seq_len) , dtype=jnp.int32)
    perturbed = tokens.at[0 , -1].set(7)

    base , changed = model(tokens) , model(perturbed)
    assert jnp.allclose(base[: , :-1 , :] , changed[: , :-1 , :] , atol=1e-5)
    assert not jnp.allclose(base[: , -1 , :] , changed[: , -1 , :] , atol=1e-5)

def test_swiglu_is_parameter_matched_with_gelu_mlp_at_gpt2_dims(tiny_config):
    from layers.ffn import MultiLayerPerceptron, SwiGLU

    config = dataclasses.replace(tiny_config , hidden_size=768 , n_heads=12)
    counts = [
        sum(p.size for p in jax.tree.leaves(nnx.state(cls(config , rngs=nnx.Rngs(0)) , nnx.Param)))
        for cls in (MultiLayerPerceptron , SwiGLU)
    ]

    # SwiGLU adds exactly one scalar
    assert counts[1] - counts[0] == 1

def test_config_rejects_unknown_variant(tiny_config):
    with pytest.raises(ValueError , match='unknown norm'):
        dataclasses.replace(tiny_config , norm='not_a_norm')

def test_config_rejects_indivisible_head_count(tiny_config):
    with pytest.raises(ValueError , match='not divisible'):
        dataclasses.replace(tiny_config , n_heads=5)
