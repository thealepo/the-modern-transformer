# benchmarks/harness.py
import json
import time
import subprocess
import statistics
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

import jax
from flax import nnx

from config import TransformerConfig
from model import Transformer


RESULTS_DIR = Path(__file__).parent / "results"


# building model and such

def build_model(config: TransformerConfig , seed: int = 0) -> Transformer:
    return Transformer(config , rngs=nnx.Rngs(seed))

def make_input(config: TransformerConfig , batch: int = 1 , seed: int = 0):
    # [batch , seq_len] of token ids
    key = jax.random.key(seed)
    return jax.random.randint(key , (batch , config.seq_len) , 0 , config.vocab_size)

# ==============================

def time_forward(model: nnx.Module , input_ids , n_runs: int = 100) -> float:
    fwd = nnx.jit(lambda m , x: m(x))    # compiled once but reused N times

    # warmup
    out = fwd(model , input_ids)
    out.block_until_ready()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = fwd(model , input_ids)
        out.block_until_ready()
        times.append(time.perf_counter() - start)

    return statistics.median(times) * 1_000.0  # return in ms

def param_count(model: nnx.Module) -> int:
    params = nnx.state(model , nnx.Param)
    return int(sum(p.size for p in jax.tree.leaves(params)))

# =====================================================

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git' , 'rev-parse' , '--short' , 'HEAD'] , text=True
        ).strip()
    except Exception:
        return 'unknown'

def _device_info() -> dict:
    d = jax.devices()[0]
    return {
        'platform': d.platform,
        'device_kind': d.device_kind,
        'jax_version': jax.__version__,
    }

def save_result(name: str , config: TransformerConfig , metrics: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True , exist_ok=True)

    record = {
        'name': name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'git_sha': _git_sha(),
        'device': _device_info(),
        'config': asdict(config),
        'metrics': metrics,
    }

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    path = RESULTS_DIR / f'{name}-{stamp}.json'
    path.write_text(json.dumps(record , indent=2))
    return path
