"""FFN slot , end-to-end: GPT-2's GELU MLP vs SwiGLU.

The two are parameter-matched on purpose: SwiGLU uses d_ff = 2/3 * (4 * hidden) ,
so 3 * 768 * 2048 == 2 * 768 * 3072 and the comparison isn't just "more params win".
Run from anywhere:  python benchmarks/bench_ffn.py [--batch 8]
"""
from harness import main_cli

CONFIGS = ['configs/baseline.yaml' , 'configs/swiglu.yaml']


if __name__ == "__main__":
    main_cli('ffn' , CONFIGS)
