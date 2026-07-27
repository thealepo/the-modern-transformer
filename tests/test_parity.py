import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from layers.norm import LayerNorm, RMSNorm

BATCH , SEQ = 2 , 8

def _x(config , scale: float = 1.0):
    return jax.random.normal(
        jax.random.key(0) , (BATCH , SEQ , config.hidden_size)
    ) * scale

def test_layernorm_matches_numpy(tiny_config):
    norm = LayerNorm(tiny_config , rngs=nnx.Rngs(0))
    x = _x(tiny_config)

    ref = np.asarray(x , dtype=np.float64)
    mean = ref.mean(axis=-1 , keepdims=True)
    var = ((ref - mean) ** 2).mean(axis=-1 , keepdims=True)
    expected = (ref - mean) / np.sqrt(var + norm.epsilon)

    np.testing.assert_allclose(np.asarray(norm(x)) , expected , rtol=1e-5 , atol=1e-5)

def test_rmsnorm_matches_numpy(tiny_config):
    norm = RMSNorm(tiny_config , rngs=nnx.Rngs(0))
    x = _x(tiny_config)

    ref = np.asarray(x , dtype=np.float64)
    expected = ref / np.sqrt((ref ** 2).mean(axis=-1 , keepdims=True) + norm.epsilon)

    np.testing.assert_allclose(np.asarray(norm(x)) , expected , rtol=1e-5 , atol=1e-5)

def test_layernorm_output_is_standardized(tiny_config):
    out = np.asarray(LayerNorm(tiny_config , rngs=nnx.Rngs(0))(_x(tiny_config , 5.0)))

    np.testing.assert_allclose(out.mean(axis=-1) , 0.0 , atol=1e-5)
    np.testing.assert_allclose(out.std(axis=-1) , 1.0 , rtol=1e-3)

def test_rmsnorm_output_has_unit_rms(tiny_config):
    out = np.asarray(RMSNorm(tiny_config , rngs=nnx.Rngs(0))(_x(tiny_config , 5.0)))
    rms = np.sqrt((out ** 2).mean(axis=-1))

    np.testing.assert_allclose(rms , 1.0 , rtol=1e-3)

def test_rmsnorm_is_scale_invariant(tiny_config):
    norm = RMSNorm(tiny_config , rngs=nnx.Rngs(0))
    x = _x(tiny_config)

    np.testing.assert_allclose(np.asarray(norm(x)) , np.asarray(norm(x * 10.0)) , rtol=1e-4 , atol=1e-4)

def test_rmsnorm_ignores_mean_but_layernorm_does_not(tiny_config):
    x = _x(tiny_config)
    shifted = x + 3.0

    layer = LayerNorm(tiny_config , rngs=nnx.Rngs(0))
    rms = RMSNorm(tiny_config , rngs=nnx.Rngs(0))

    np.testing.assert_allclose(np.asarray(layer(x)) , np.asarray(layer(shifted)) ,
                               rtol=1e-4 , atol=1e-4)
    assert not jnp.allclose(rms(x) , rms(shifted) , atol=1e-3)
