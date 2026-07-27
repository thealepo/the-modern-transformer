import json
import math
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

import registry
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
    n_warmup: int = 3
    min_block_ms: float = 100.0

    def __post_init__(self):
        for field in ('batch' , 'n_runs' , 'n_reps' , 'n_warmup'):
            if getattr(self , field) < 1:
                raise ValueError(f'{field} must be >= 1 , got {getattr(self , field)}')
        if self.min_block_ms <= 0:
            raise ValueError(f'min_block_ms must be > 0 , got {self.min_block_ms}')

# configs ======

def resolve_config(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    from_root = ROOT / path
    return from_root if from_root.exists() else CONFIGS_DIR / path

def load_config(config_path: str | Path) -> TransformerConfig:
    return TransformerConfig.from_yaml(resolve_config(config_path))

def build_model(config: TransformerConfig , seed: int = 0) -> Transformer:
    return Transformer(config , rngs=nnx.Rngs(seed))

def make_tokens(config: TransformerConfig , batch: int , seed: int = 0):
    # [batch , seq_len] of token ids
    return jax.random.randint(
        jax.random.key(seed) , (batch , config.seq_len) , 0 , config.vocab_size
    )

def make_activations(config: TransformerConfig , batch: int , seed: int = 0):
    # [batch , seq_len , hidden_size] -- what every slot component receives
    return jax.random.normal(
        jax.random.key(seed) , (batch , config.seq_len , config.hidden_size)
    )

def param_count(model: nnx.Module) -> int:
    return int(sum(p.size for p in jax.tree.leaves(nnx.state(model , nnx.Param))))

# compilation ====

def forward_fn(model: nnx.Module):
    graphdef , state = nnx.split(model)

    @jax.jit
    def forward(state , tokens):
        return nnx.merge(graphdef , state)(tokens)  # [batch , seq_len , vocab_size]

    return forward , state

def train_step_fn(model: nnx.Module , learning_rate: float = 1e-3):
    graphdef , state = nnx.split(model)

    def loss_fn(state , tokens):
        logits = nnx.merge(graphdef , state)(tokens)   # [batch , seq_len , vocab_size]
        # Real next-token objective: predict tokens[t+1] from position t.
        logits , targets = logits[: , :-1 , :] , tokens[: , 1:]
        log_probs = jax.nn.log_softmax(logits , axis=-1)
        return -jnp.take_along_axis(log_probs , targets[... , None] , axis=-1).mean()

    @jax.jit(donate_argnums=0)
    def step(state , tokens):
        grads = jax.grad(loss_fn)(state , tokens)
        return jax.tree.map(lambda p , g: p - learning_rate * g , state , grads)

    return step , state

def component_fn(slot: str , config: TransformerConfig , seed: int = 0):
    component = registry.resolve(slot , getattr(config , slot))(config , rngs=nnx.Rngs(seed))
    graphdef , state = nnx.split(component)

    @jax.jit
    def call(state , x):
        return nnx.merge(graphdef , state)(x)

    return call , state , param_count(component)

def _stats(per_call_ms: list[float]) -> dict:
    return {
        'ms': statistics.median(per_call_ms),
        'min_ms': min(per_call_ms),
        'max_ms': max(per_call_ms),
        'stdev_ms': statistics.stdev(per_call_ms) if len(per_call_ms) > 1 else 0.0,
    }

def _calls_per_block(fn , *args , spec: BenchSpec) -> int:
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    per_call_ms = (time.perf_counter() - start) * 1e3

    if per_call_ms <= 0:
        return spec.n_runs
    return max(spec.n_runs , math.ceil(spec.min_block_ms / per_call_ms))

def time_calls(fn , *args , spec: BenchSpec) -> dict:
    for _ in range(spec.n_warmup):
        jax.block_until_ready(fn(*args))

    n_runs = _calls_per_block(fn , *args , spec=spec)

    per_call = []
    for _ in range(spec.n_reps):
        start = time.perf_counter()
        for _ in range(n_runs):
            out = fn(*args)
        jax.block_until_ready(out)
        per_call.append((time.perf_counter() - start) / n_runs * 1e3)
    return _stats(per_call) | {'n_runs': n_runs}

def time_train(step , state , tokens , spec: BenchSpec) -> dict:
    for _ in range(spec.n_warmup):
        state = step(state , tokens)
    jax.block_until_ready(state)

    per_call = []
    for _ in range(spec.n_reps):
        start = time.perf_counter()
        for _ in range(spec.n_runs):
            state = step(state , tokens)
        jax.block_until_ready(state)
        per_call.append((time.perf_counter() - start) / spec.n_runs * 1e3)
    return _stats(per_call) | {'n_runs': spec.n_runs}

# ============== cost

def flops_per_token(model: nnx.Module , config: TransformerConfig) -> float:
    matmul_params = sum(
        int(node.kernel.size)
        for _ , node in nnx.iter_graph(model) if isinstance(node , nnx.Linear)
    )
    matmul_params += int(model.wte.embedding.size)

    attention = 2 * (2 * config.seq_len * config.hidden_size) * config.n_layers
    return 2.0 * matmul_params + attention

def compile_and_measure(fn , *args) -> dict:
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    compile_ms = (time.perf_counter() - start) * 1e3

    mem = compiled.memory_analysis()
    metrics = {'compile_ms': compile_ms}
    if mem is not None:
        metrics |= {
            'params_mib': mem.argument_size_in_bytes / 2**20,
            'output_mib': mem.output_size_in_bytes / 2**20,
            'temp_mib': mem.temp_size_in_bytes / 2**20,
        }
    return metrics

def peak_memory_mib() -> float | None:
    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get('peak_bytes_in_use')
    return None if peak is None else peak / 2**20

def measure_component(slot: str , config: TransformerConfig , spec: BenchSpec) -> dict:
    call , state , n_params = component_fn(slot , config , spec.seed)
    x = make_activations(config , spec.batch , spec.seed)

    metrics = {f'micro_{k}': v for k , v in compile_and_measure(call , state , x).items()}
    metrics |= {f'micro_{k}': v for k , v in time_calls(call , state , x , spec=spec).items()}
    metrics['micro_param_count'] = n_params
    return metrics

def measure_forward(config: TransformerConfig , spec: BenchSpec) -> dict:
    model = build_model(config , seed=spec.seed)
    tokens = make_tokens(config , spec.batch , spec.seed)
    forward , state = forward_fn(model)

    metrics = {
        'param_count': param_count(model),
        'flops_per_token': flops_per_token(model , config),
    }
    metrics |= {f'fwd_{k}': v for k , v in compile_and_measure(forward , state , tokens).items()}
    metrics |= {f'fwd_{k}': v for k , v in time_calls(forward , state , tokens , spec=spec).items()}
    metrics['fwd_tokens_per_sec'] = tokens.size / (metrics['fwd_ms'] / 1e3)
    return metrics

def measure_train(config: TransformerConfig , spec: BenchSpec) -> dict:
    tokens = make_tokens(config , spec.batch , spec.seed)
    step , state = train_step_fn(build_model(config , seed=spec.seed))

    metrics = {f'step_{k}': v for k , v in compile_and_measure(step , state , tokens).items()}
    metrics |= {f'step_{k}': v for k , v in time_train(step , state , tokens , spec).items()}
    metrics['step_tokens_per_sec'] = tokens.size / (metrics['step_ms'] / 1e3)
    return metrics

def bench_config(config_path: str , slot: str , spec: BenchSpec) -> dict:
    config = load_config(config_path)

    metrics = measure_component(slot , config , spec)
    metrics |= measure_forward(config , spec)
    metrics |= measure_train(config , spec)          # peak memory lives here
    metrics['peak_mib'] = peak_memory_mib()

    variant = getattr(config , slot)
    path = save_result(f'{slot}_{variant}' , config , spec , metrics)
    return {'slot': slot , 'variant': variant , 'config_path': str(config_path),
            'metrics': metrics , 'result_path': str(path)}

# ===========

def _git(*args: str , default: str = 'unknown') -> str:
    try:
        return subprocess.check_output(['git' , *args] , cwd=ROOT , text=True).strip()
    except Exception:
        return default

def _environment() -> dict:
    device = jax.devices()[0]
    return {
        'platform': device.platform,
        'device_kind': device.device_kind,
        'n_devices': jax.device_count(),
        'jax_version': jax.__version__,
        'git_sha': _git('rev-parse' , '--short' , 'HEAD'),
        'git_dirty': bool(_git('status' , '--porcelain' , default='')),
    }

def save_result(name: str , config: TransformerConfig , spec: BenchSpec , metrics: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True , exist_ok=True)
    stamp = datetime.now(timezone.utc)

    record = {
        'name': name,
        'timestamp': stamp.isoformat(),
        'environment': _environment(),
        'config': asdict(config),
        'spec': asdict(spec),
        'metrics': metrics,
    }
    path = RESULTS_DIR / f'{name}-{stamp.strftime("%Y%m%d-%H%M%S")}.json'
    path.write_text(json.dumps(record , indent=2))
    return path

def emit(record: dict) -> None:
    print(_RESULT_LINE + json.dumps(record) , flush=True)

def run_isolated(config_path: str , extra_args: list[str]) -> dict:
    # argv[0] must be absolute: the child runs with cwd=ROOT , so a relative script path
    # would be re-resolved against ROOT instead of the cwd we were invoked from.
    script = str(Path(sys.argv[0]).resolve())
    proc = subprocess.run(
        [sys.executable , script , '--config' , config_path , *extra_args],
        cwd=ROOT , capture_output=True , text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_LINE):
            return json.loads(line[len(_RESULT_LINE):])
    raise RuntimeError(
        f'benchmark subprocess for {config_path} produced no result '
        f'(exit {proc.returncode})\n--- stderr ---\n{proc.stderr[-2000:]}'
    )

# ============

def check_single_slot_diff(slot: str , config_paths: list[str]) -> None:
    base_path , *others = config_paths
    base = asdict(load_config(base_path))

    for path in others:
        variant = asdict(load_config(path))

        drift = {k: (base[k] , variant[k]) for k in base
                 if k != slot and base[k] != variant[k]}
        if drift:
            fields = ' , '.join(f'{k}: {b!r} -> {v!r}' for k , (b , v) in drift.items())
            raise ValueError(
                f'{path} differs from {base_path} outside the {slot!r} slot ({fields}); '
                f'a measured delta would not be attributable to {slot}'
            )
        if base[slot] == variant[slot]:
            raise ValueError(
                f'{path} has the same {slot}={variant[slot]!r} as {base_path}; '
                f'there is nothing to compare'
            )

_COLUMNS = [
    ('micro ms'   , lambda m: f"{m['micro_ms']:.4f}"),
    ('fwd ms'     , lambda m: f"{m['fwd_ms']:.3f}"),
    ('fwd+bwd ms' , lambda m: f"{m['step_ms']:.3f}"),
    ('tok/s'      , lambda m: f"{m['step_tokens_per_sec']:,.0f}"),
    ('peak MiB'   , lambda m: f"{m['peak_mib']:.0f}"),
    ('MFLOP/tok'  , lambda m: f"{m['flops_per_token'] / 1e6:.1f}"),
    ('params'     , lambda m: f"{m['param_count']:,d}"),
]
_DELTAS = [('micro' , 'micro_ms') , ('fwd' , 'fwd_ms') , ('fwd+bwd' , 'step_ms'),
           ('peak' , 'peak_mib') , ('params' , 'param_count')]

def _cell(metrics: dict , render) -> str:
    try:
        return render(metrics)
    except (KeyError , TypeError):
        return 'n/a'

def print_table(slot: str , rows: list[dict] , spec: BenchSpec) -> None:
    labels = [row['variant'] for row in rows]
    cells = [[_cell(row['metrics'] , render) for _ , render in _COLUMNS] for row in rows]

    label_w = max([len(slot) , *(len(label) for label in labels)])
    widths = [max([len(header) , *(len(row[i]) for row in cells)])
              for i , (header , _) in enumerate(_COLUMNS)]

    used = {f'{name} {row["metrics"].get(f"{name}_n_runs" , "?")}'
            for row in rows for name in ('micro' , 'fwd' , 'step')}
    print(f'\nbatch={spec.batch}  seed={spec.seed}  warmup={spec.n_warmup}  '
          f'{spec.n_reps} blocks of [{" , ".join(sorted(used))}] calls')
    header = f'{slot:>{label_w}} ' + ' '.join(
        f'{name:>{w}}' for (name , _) , w in zip(_COLUMNS , widths))
    print(header)
    print('-' * len(header))
    for label , row in zip(labels , cells):
        print(f'{label:>{label_w}} ' + ' '.join(f'{c:>{w}}' for c , w in zip(row , widths)))

    if len(rows) < 2:
        return

    base = rows[0]['metrics']
    print('-' * len(header))
    for row in rows[1:]:
        parts = []
        for name , key in _DELTAS:
            before , after = base.get(key) , row['metrics'].get(key)
            if before and after is not None:
                parts.append(f'{name} {100 * (after - before) / before:+.2f}%')
        print(f"{row['variant']:>{label_w}}  vs {rows[0]['variant']}:  " + '   '.join(parts))

    print('\n(negative = the variant is faster / smaller. `micro ms` is the component\n'
          ' alone; compare it against `fwd ms` to see how much of the component delta\n'
          ' survives into the whole model.)')

def _median_row(rows: list[dict]) -> dict:
    return sorted(rows , key=lambda r: r['metrics']['fwd_ms'])[len(rows) // 2]

def main_cli(slot: str , config_paths: list[str]) -> None:
    import argparse

    defaults = BenchSpec()
    parser = argparse.ArgumentParser(description=f'{slot} slot: ' + ' vs '.join(config_paths))
    parser.add_argument('--config' , help='internal: measure only this config , emit JSON')
    parser.add_argument('--batch' , type=int , default=defaults.batch)
    parser.add_argument('--n-runs' , type=int , default=defaults.n_runs)
    parser.add_argument('--n-reps' , type=int , default=defaults.n_reps)
    parser.add_argument('--n-warmup' , type=int , default=defaults.n_warmup)
    parser.add_argument('--seed' , type=int , default=defaults.seed)
    parser.add_argument('--rounds' , type=int , default=3 ,
                        help='interleaved passes over every config; odd , so the median '
                             'of each variant is a real round')
    args = parser.parse_args()

    spec = BenchSpec(batch=args.batch , seed=args.seed , n_runs=args.n_runs ,
                     n_reps=args.n_reps , n_warmup=args.n_warmup)

    if args.config:
        emit(bench_config(args.config , slot , spec))
        return

    check_single_slot_diff(slot , config_paths)

    forwarded = ['--batch' , str(spec.batch) , '--n-runs' , str(spec.n_runs),
                 '--n-reps' , str(spec.n_reps) , '--n-warmup' , str(spec.n_warmup),
                 '--seed' , str(spec.seed)]

    per_config: dict[str , list[dict]] = {path: [] for path in config_paths}
    for round_idx in range(args.rounds):
        for path in config_paths:
            row = run_isolated(path , forwarded)
            per_config[path].append(row)
            print(f"  round {round_idx}  {row['variant']:>12}  "
                  f"micro {row['metrics']['micro_ms']:7.4f} ms  "
                  f"fwd {row['metrics']['fwd_ms']:7.3f} ms  "
                  f"fwd+bwd {row['metrics']['step_ms']:8.3f} ms" , flush=True)

    if args.rounds > 1:
        for rounds in per_config.values():
            parts = []
            for name , key in (('micro' , 'micro_ms') , ('fwd' , 'fwd_ms') , ('fwd+bwd' , 'step_ms')):
                values = [r['metrics'][key] for r in rounds]
                spread = max(values) - min(values)
                parts.append(f'{name} {spread:.4f} ms ({100 * spread / statistics.median(values):.1f}%)')
            print(f"  {rounds[0]['variant']:>12}  between-process spread:  " + '   '.join(parts))

    print_table(slot , [_median_row(per_config[p]) for p in config_paths] , spec)
