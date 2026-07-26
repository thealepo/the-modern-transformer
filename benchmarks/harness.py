import json
import sys
import time
import subprocess
import statistics
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0 , str(ROOT / 'src'))

import jax
import jax.numpy as jnp
from flax import nnx

from config import TransformerConfig
from model import Transformer

RESULTS_DIR = Path(__file__).parent / 'results'
CONFIGS_DIR = ROOT / 'configs'
_RESULT_LINE = '###RESULT###'


@dataclass(frozen=True , kw_only=True , slots=True)
class BenchSpec:
    batch: int = 8
    seed: int = 0
    n_runs: int = 50
    n_reps: int = 5
    n_warmup: int = 2


# =====================

def build_model(config: TransformerConfig , seed: int = 0) -> Transformer:
    return Transformer(config , rngs=nnx.Rngs(seed))

def make_input(config: TransformerConfig , batch: int = 1 , seed: int = 0):
    # [batch , seq_len] of token ids
    key = jax.random.key(seed)
    return jax.random.randint(key , (batch , config.seq_len) , 0 , config.vocab_size)

def param_count(model: nnx.Module) -> int:
    params = nnx.state(model , nnx.Param)
    return int(sum(p.size for p in jax.tree.leaves(params)))


# =========================

def compile_forward(model: nnx.Module):
    graphdef , state = nnx.split(model)

    @jax.jit
    def forward(state , input_ids):
        return nnx.merge(graphdef , state)(input_ids)  # [batch , seq_len , vocab_size]

    return forward , state

def compile_train_step(model: nnx.Module , learning_rate: float = 1e-3):
    graphdef , state = nnx.split(model)

    def loss_fn(state , input_ids , labels):
        logits = nnx.merge(graphdef , state)(input_ids)  # [batch , seq_len , vocab_size]
        log_probs = jax.nn.log_softmax(logits , axis=-1)
        return -jnp.take_along_axis(log_probs , labels[... , None] , axis=-1).mean()

    @jax.jit(donate_argnums=0)  # donating state
    def step(state , input_ids , labels):
        grads = jax.grad(loss_fn)(state , input_ids , labels)
        return jax.tree.map(lambda p , g: p - learning_rate * g , state , grads)

    return step , state

# ======

def _stats(per_iter_ms: list[float]) -> dict:
    return {
        'latency_ms': statistics.median(per_iter_ms),
        'min_ms': min(per_iter_ms),
        'max_ms': max(per_iter_ms),
        'stdev_ms': statistics.stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0,
    }

def time_forward(forward , state , input_ids , spec: BenchSpec = BenchSpec()) -> dict:
    for _ in range(spec.n_warmup):
        jax.block_until_ready(forward(state , input_ids))

    per_iter = []
    for _ in range(spec.n_reps):
        start = time.perf_counter()
        for _ in range(spec.n_runs):
            out = forward(state , input_ids)
        jax.block_until_ready(out)
        per_iter.append((time.perf_counter() - start) / spec.n_runs * 1_000.0)

    blocking = []
    for _ in range(spec.n_runs):
        start = time.perf_counter()
        jax.block_until_ready(forward(state , input_ids))
        blocking.append((time.perf_counter() - start) * 1_000.0)

    metrics = _stats(per_iter)
    metrics['blocking_latency_ms'] = statistics.median(blocking)
    metrics['host_overhead_ms'] = metrics['blocking_latency_ms'] - metrics['latency_ms']
    metrics['tokens_per_sec'] = (
        input_ids.shape[0] * input_ids.shape[1] / (metrics['latency_ms'] / 1_000.0)
    )
    return metrics

def time_train_step(step , state , input_ids , labels , spec: BenchSpec = BenchSpec()) -> dict:
    """Same idea for fwd+bwd. State is threaded because the buffers are donated."""
    for _ in range(spec.n_warmup):
        state = step(state , input_ids , labels)
    jax.block_until_ready(state)

    per_iter = []
    for _ in range(spec.n_reps):
        start = time.perf_counter()
        for _ in range(spec.n_runs):
            state = step(state , input_ids , labels)
        jax.block_until_ready(state)
        per_iter.append((time.perf_counter() - start) / spec.n_runs * 1_000.0)

    metrics = {f'train_{k}': v for k , v in _stats(per_iter).items()}
    metrics['train_tokens_per_sec'] = (
        input_ids.shape[0] * input_ids.shape[1] / (metrics['train_latency_ms'] / 1_000.0)
    )
    return metrics


# ==================================== FLOP stuff

def flops_per_token(model: nnx.Module , config: TransformerConfig) -> float:
    matmul_params = param_count(model) - param_count(model.pos)
    attention = 4 * config.n_layers * config.seq_len * config.hidden_size
    return 2.0 * matmul_params + attention

def flops_per_token_xla(forward , state , input_ids) -> float | None:
    analysis = forward.lower(state , input_ids).compile().cost_analysis()
    if isinstance(analysis , list):
        analysis = analysis[0]
    flops = (analysis or {}).get('flops')
    if flops is None:
        return None
    return float(flops) / (input_ids.shape[0] * input_ids.shape[1])

def peak_memory_bytes() -> int | None:
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get('peak_bytes_in_use')

# =============

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git' , 'rev-parse' , '--short' , 'HEAD'] , cwd=ROOT , text=True
        ).strip()
    except Exception:
        return 'unknown'

def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ['git' , 'status' , '--porcelain'] , cwd=ROOT , text=True
        ).strip())
    except Exception:
        return False

def _device_info() -> dict:
    d = jax.devices()[0]
    return {
        'platform': d.platform,
        'device_kind': d.device_kind,
        'n_devices': jax.device_count(),
        'jax_version': jax.__version__,
    }

def save_result(name: str , config: TransformerConfig , spec: BenchSpec , metrics: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True , exist_ok=True)

    record = {
        'name': name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'git_sha': _git_sha(),
        'git_dirty': _git_dirty(),
        'device': _device_info(),
        'config': asdict(config),
        'spec': asdict(spec),
        'metrics': metrics,
    }

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    path = RESULTS_DIR / f'{name}-{stamp}.json'
    path.write_text(json.dumps(record , indent=2))
    return path

# ===========

def emit(record: dict) -> None:
    print(_RESULT_LINE + json.dumps(record) , flush=True)

def run_isolated(config_path: str , extra_args: list[str] | None = None) -> dict:
    proc = subprocess.run(
        [sys.executable , sys.argv[0] , '--config' , config_path , *(extra_args or [])],
        cwd=ROOT , capture_output=True , text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_LINE):
            return json.loads(line[len(_RESULT_LINE):])
    raise RuntimeError(
        f"benchmark subprocess for {config_path} produced no result "
        f"(exit {proc.returncode})\n--- stderr ---\n{proc.stderr[-2000:]}"
    )

def bench_config(config_path: str , slot: str , spec: BenchSpec) -> dict:
    config = TransformerConfig.from_yaml(CONFIGS_DIR / Path(config_path).name)
    model = build_model(config , seed=spec.seed)
    variant = getattr(config , slot)

    n_params = param_count(model)

    x = make_input(config , batch=spec.batch , seed=spec.seed)
    y = make_input(config , batch=spec.batch , seed=spec.seed + 1)

    forward , state = compile_forward(model)
    metrics = time_forward(forward , state , x , spec)
    metrics['param_count'] = n_params
    metrics['flops_per_token'] = flops_per_token(model , config)
    metrics['flops_per_token_xla'] = flops_per_token_xla(forward , state , x)

    step , train_state = compile_train_step(model)
    metrics |= time_train_step(step , train_state , x , y , spec)

    metrics['peak_memory_mib'] = (peak_memory_bytes() or 0) / 2**20

    path = save_result(f'{slot}_{variant}' , config , spec , metrics)
    return {'slot': slot , 'variant': variant , 'config_path': config_path,
            'metrics': metrics , 'result_path': str(path)}


# COMPARISON CLI
_COLUMNS = [
    ('fwd ms'     , 'latency_ms'       , 10 , '.3f' , 1.0),
    ('tok/s'      , 'tokens_per_sec'   , 10 , '.0f' , 1.0),
    ('host ms'    , 'host_overhead_ms' ,  8 , '.3f' , 1.0),
    ('fwd+bwd ms' , 'train_latency_ms' , 11 , '.3f' , 1.0),
    ('peak MiB'   , 'peak_memory_mib'  ,  9 , '.0f' , 1.0),
    ('MFLOP/tok'  , 'flops_per_token'  , 10 , '.1f' , 1e-6),
    ('params'     , 'param_count'      , 13 , ',d'  , 1.0),
]

def _cell(metrics: dict , key: str , width: int , fmt: str , scale: float) -> str:
    value = metrics.get(key)
    if value is None:
        return f"{'n/a':>{width}}"
    if scale != 1.0:
        value = value * scale
    return f"{value:>{width}{fmt}}"

def print_table(slot: str , rows: list[dict] , spec: BenchSpec) -> None:
    header = f"{slot:>12} " + ' '.join(
        f"{name:>{width}}" for name , _ , width , _ , _ in _COLUMNS
    )
    print(f"\nbatch={spec.batch}  n_runs={spec.n_runs}x{spec.n_reps}  seed={spec.seed}")
    print(header)
    print('-' * len(header))

    for row in rows:
        cells = ' '.join(
            _cell(row['metrics'] , key , width , fmt , scale)
            for _ , key , width , fmt , scale in _COLUMNS
        )
        print(f"{row['variant']:>12} {cells}")

    if len(rows) > 1:
        base = rows[0]['metrics']
        print('-' * len(header))
        for row in rows[1:]:
            parts = []
            for name , key in (('fwd' , 'latency_ms') , ('fwd+bwd' , 'train_latency_ms'),
                               ('peak' , 'peak_memory_mib') , ('params' , 'param_count')):
                b , v = base.get(key) , row['metrics'].get(key)
                if b:
                    parts.append(f"{name} {100 * (v - b) / b:+.2f}%")
            print(f"{row['variant']:>12}  vs {rows[0]['variant']}:  " + '   '.join(parts))
        print("\n(negative = the variant is faster / smaller. `host ms` is Python dispatch\n"
              " cost excluded from `fwd ms`; if it rivals `fwd ms` you are host-bound.)")

def _median_row(rows: list[dict]) -> dict:
    """The round whose forward latency is the median. A whole row , not a blend ,
    so every metric in the reported line came from the same consistent process."""
    ordered = sorted(rows , key=lambda r: r['metrics']['latency_ms'])
    return ordered[len(ordered) // 2]

def main_cli(slot: str , config_paths: list[str]) -> None:
    import argparse

    defaults = BenchSpec()
    parser = argparse.ArgumentParser(description=f'{slot} slot: ' + ' vs '.join(config_paths))
    parser.add_argument('--config' , help='internal: measure only this config')
    parser.add_argument('--batch' , type=int , default=defaults.batch)
    parser.add_argument('--n-runs' , type=int , default=defaults.n_runs)
    parser.add_argument('--n-reps' , type=int , default=defaults.n_reps)
    parser.add_argument('--seed' , type=int , default=defaults.seed)
    parser.add_argument('--rounds' , type=int , default=3 ,
                        help='interleaved passes over every config (odd number)')
    args = parser.parse_args()

    spec = BenchSpec(batch=args.batch , seed=args.seed ,
                     n_runs=args.n_runs , n_reps=args.n_reps)

    if args.config:
        emit(bench_config(args.config , slot , spec))
        return

    forwarded = ['--batch' , str(spec.batch) , '--n-runs' , str(spec.n_runs),
                 '--n-reps' , str(spec.n_reps) , '--seed' , str(spec.seed)]

    per_variant: dict[str , list[dict]] = {}
    for round_idx in range(args.rounds):
        for path in config_paths:
            row = run_isolated(path , forwarded)
            per_variant.setdefault(row['variant'] , []).append(row)
            print(f"  round {round_idx}  {row['variant']:>12}  "
                  f"fwd {row['metrics']['latency_ms']:7.3f} ms  "
                  f"fwd+bwd {row['metrics']['train_latency_ms']:8.3f} ms" , flush=True)

    for variant , rounds in per_variant.items():
        spread = max(r['metrics']['latency_ms'] for r in rounds) \
               - min(r['metrics']['latency_ms'] for r in rounds)
        base = _median_row(rounds)['metrics']['latency_ms']
        print(f"  {variant:>12}  between-process spread {spread:6.3f} ms ({100 * spread / base:4.1f}%)")

    rows = [_median_row(per_variant[v]) for v in per_variant]
    print_table(slot , rows , spec)
