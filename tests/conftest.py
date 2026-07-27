import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

for subdir in ('src' , 'benchmarks'):
    path = str(ROOT / subdir)
    if path not in sys.path:
        sys.path.insert(0 , path)
TINY = dict(vocab_size=256 , seq_len=32 , hidden_size=64 , n_heads=4 , n_layers=2 ,
            norm='layernorm' , positional='learned' , ffn='gelu_mlp' , attention='mha')

@pytest.fixture
def tiny_config():
    from config import TransformerConfig
    return TransformerConfig(**TINY)

@pytest.fixture
def write_config(tmp_path):
    import yaml

    def _write(name: str , **overrides) -> Path:
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(TINY | overrides))
        return path

    return _write
