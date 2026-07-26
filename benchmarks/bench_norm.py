"""Norm slot , end-to-end: GPT-2's LayerNorm vs RMSNorm.

Everything except `norm:` is identical between the two configs.
Run from anywhere:  python benchmarks/bench_norm.py [--batch 8]
"""
from harness import main_cli

CONFIGS = ['configs/baseline.yaml' , 'configs/rmsnorm.yaml']


if __name__ == "__main__":
    main_cli('norm' , CONFIGS)
