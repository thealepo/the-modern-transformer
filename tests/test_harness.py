"""Harness tests. Most of these are regressions for bugs that were live in the old
harness -- a benchmark you can't trust is worse than no benchmark.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import harness
from harness import BenchSpec

FAST = BenchSpec(batch=2 , n_runs=2 , n_reps=2 , n_warmup=1)


def test_resolve_config_accepts_root_relative_and_bare_names():
    assert harness.resolve_config('configs/baseline.yaml') == harness.CONFIGS_DIR / 'baseline.yaml'
    assert harness.resolve_config('baseline.yaml') == harness.CONFIGS_DIR / 'baseline.yaml'

def test_resolve_config_keeps_absolute_paths(write_config):
    path = write_config('somewhere')
    assert harness.resolve_config(path) == path

def test_resolve_config_does_not_smuggle_a_stray_path_into_configs():
    resolved = harness.resolve_config('other/dir/baseline.yaml')
    assert resolved != harness.CONFIGS_DIR / 'baseline.yaml'
    assert not resolved.exists()

def test_check_single_slot_diff_accepts_a_one_field_change(write_config):
    harness.check_single_slot_diff(
        'norm' , [write_config('base') , write_config('variant' , norm='rmsnorm')])

def test_check_single_slot_diff_rejects_drift_outside_the_slot(write_config):
    drifted = write_config('drifted' , norm='rmsnorm' , n_layers=4)
    with pytest.raises(ValueError , match='outside the .norm. slot'):
        harness.check_single_slot_diff('norm' , [write_config('base') , drifted])

def test_check_single_slot_diff_rejects_two_configs_with_the_same_variant(write_config):
    with pytest.raises(ValueError , match='nothing to compare'):
        harness.check_single_slot_diff('norm' , [write_config('a') , write_config('b')])

def test_real_configs_are_single_slot_diffs_off_baseline():
    harness.check_single_slot_diff('norm' , ['configs/baseline.yaml' , 'configs/rmsnorm.yaml'])
    harness.check_single_slot_diff('ffn' , ['configs/baseline.yaml' , 'configs/swiglu.yaml'])

def test_flops_per_token_counts_only_matmuls(tiny_config):
    model = harness.build_model(tiny_config)
    c = tiny_config

    per_layer = 4 * c.hidden_size ** 2 + 2 * c.hidden_size * c.mlp_hidden_size
    matmul_params = c.n_layers * per_layer + c.vocab_size * c.hidden_size
    attention = 2 * (2 * c.seq_len * c.hidden_size) * c.n_layers

    assert harness.flops_per_token(model , c) == 2.0 * matmul_params + attention


def test_flops_per_token_excludes_elementwise_params(tiny_config):
    import dataclasses

    layer_cfg = tiny_config
    rms_cfg = dataclasses.replace(tiny_config , norm='rmsnorm')

    layer_flops = harness.flops_per_token(harness.build_model(layer_cfg) , layer_cfg)
    rms_flops = harness.flops_per_token(harness.build_model(rms_cfg) , rms_cfg)

    assert layer_flops == rms_flops
    # ... even though the parameter counts genuinely differ.
    assert harness.param_count(harness.build_model(layer_cfg)) \
         > harness.param_count(harness.build_model(rms_cfg))

def test_flops_per_token_ignores_positional_embeddings(tiny_config):
    model = harness.build_model(tiny_config)
    flops = harness.flops_per_token(model , tiny_config)

    assert flops < 2.0 * harness.param_count(model) + 2 * (
        2 * tiny_config.seq_len * tiny_config.hidden_size) * tiny_config.n_layers

def test_time_calls_reports_positive_latencies(tiny_config):
    forward , state = harness.forward_fn(harness.build_model(tiny_config))
    tokens = harness.make_tokens(tiny_config , FAST.batch)

    stats = harness.time_calls(forward , state , tokens , spec=FAST)

    assert stats['ms'] > 0
    assert stats['min_ms'] <= stats['ms'] <= stats['max_ms']


def test_autoranging_gives_a_fast_component_more_calls_per_block(tiny_config):
    spec = BenchSpec(batch=2 , n_runs=2 , n_reps=2 , n_warmup=1 , min_block_ms=20.0)
    call , state , _ = harness.component_fn('norm' , tiny_config)
    x = harness.make_activations(tiny_config , spec.batch)

    stats = harness.time_calls(call , state , x , spec=spec)

    assert stats['n_runs'] > spec.n_runs
    assert stats['ms'] * stats['n_runs'] >= spec.min_block_ms * 0.5


def test_autoranging_never_reduces_the_requested_call_count(tiny_config):
    spec = BenchSpec(batch=2 , n_runs=4 , n_reps=2 , n_warmup=1 , min_block_ms=1e-6)
    forward , state = harness.forward_fn(harness.build_model(tiny_config))
    tokens = harness.make_tokens(tiny_config , spec.batch)

    assert harness.time_calls(forward , state , tokens , spec=spec)['n_runs'] == 4


def test_bench_spec_rejects_degenerate_counts():
    with pytest.raises(ValueError , match='n_runs must be >= 1'):
        BenchSpec(n_runs=0)


def test_train_step_actually_updates_parameters(tiny_config):
    step , state = harness.train_step_fn(harness.build_model(tiny_config))
    tokens = harness.make_tokens(tiny_config , FAST.batch)

    before = jax.tree.leaves(state)[0].copy()
    after = jax.tree.leaves(step(state , tokens))[0]

    assert not jnp.allclose(before , after)

def test_train_step_donation_does_not_outlive_its_phase(tiny_config):
    train = harness.measure_train(tiny_config , FAST)
    forward = harness.measure_forward(tiny_config , FAST)  # after training , on purpose

    assert train['step_ms'] > 0
    assert forward['fwd_ms'] > 0

def test_measure_component_matches_the_configured_variant(tiny_config):
    import dataclasses

    gelu = harness.measure_component('ffn' , tiny_config , FAST)
    swiglu = harness.measure_component(
        'ffn' , dataclasses.replace(tiny_config , ffn='swiglu_mlp') , FAST)

    assert gelu['micro_ms'] > 0 and swiglu['micro_ms'] > 0
    # Roughly parameter-matched by construction (d_ff = 2/3 * 4 * hidden). Exact
    # matching is pinned down at GPT-2 dims in
    # test_shapes.test_swiglu_is_parameter_matched_with_gelu_mlp_at_gpt2_dims -- at the
    # tiny dims used here , int() truncation of d_ff costs a few hundred parameters.
    assert swiglu['micro_param_count'] == pytest.approx(gelu['micro_param_count'] , rel=0.01)

# =====

def test_bench_config_produces_every_reported_metric(write_config):
    row = harness.bench_config(str(write_config('e2e')) , 'norm' , FAST)

    assert row['variant'] == 'layernorm'
    for key in ('micro_ms' , 'fwd_ms' , 'step_ms' , 'fwd_tokens_per_sec' ,
                'step_tokens_per_sec' , 'param_count' , 'flops_per_token'):
        assert row['metrics'][key] > 0 , key

    saved = json.loads(Path(row['result_path']).read_text())
    assert saved['config']['norm'] == 'layernorm'
    assert saved['environment']['jax_version'] == jax.__version__


def test_every_table_column_renders(write_config):
    row = {'variant': 'layernorm' , 'metrics': {'micro_ms': 1.0 , 'fwd_ms': 2.0 ,
                                                'step_ms': 3.0 , 'step_tokens_per_sec': 4.0 ,
                                                'peak_mib': None , 'flops_per_token': 5e6}}
    harness.print_table('norm' , [row] , FAST)   # param_count absent , peak_mib None


def test_run_isolated_works_from_any_cwd(tmp_path , write_config):
    script = tmp_path / 'bench_tmp.py'
    script.write_text(textwrap.dedent(f'''
        import sys
        sys.path.insert(0 , {str(harness.ROOT / "benchmarks")!r})
        from harness import main_cli
        main_cli('norm' , [{str(write_config("base"))!r} ,
                           {str(write_config("variant" , norm="rmsnorm"))!r}])
    '''))

    proc = subprocess.run(
        [sys.executable , str(script) , '--rounds' , '1' , '--batch' , '2' ,
         '--n-runs' , '2' , '--n-reps' , '1' , '--n-warmup' , '1'],
        cwd=tmp_path , capture_output=True , text=True , timeout=900,
    )

    assert proc.returncode == 0 , proc.stderr[-3000:]
    assert 'layernorm' in proc.stdout and 'rmsnorm' in proc.stdout
