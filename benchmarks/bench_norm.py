from config import TransformerConfig
from harness import build_model , make_input , time_forward , param_count , save_result

CONFIGS = ['configs/baseline.yaml' , 'configs/rmsnorm.yaml']


def main():
    print(f"{'norm':12} {'latency (ms)':>14} {'tokens/sec':>14} {'params':>10}")
    print("-" * 54)

    for cfg_path in CONFIGS:
        config = TransformerConfig.from_yaml(cfg_path)
        model = build_model(config)
        x = make_input(config)  # [batch , seq_len]

        latency_ms = time_forward(model , x)
        tokens_per_sec = (x.shape[0] * x.shape[1]) / (latency_ms / 1000)

        metrics = {
            'latency_ms': latency_ms,
            'tokens_per_sec': tokens_per_sec,
            'param_count': param_count(model),
        }
        save_result(f'norm_{config.norm}' , config , metrics)

        print(
            f"{config.norm:12} {latency_ms:14.3f} {tokens_per_sec:14.1f} "
            f"{metrics['param_count']:10d}"
        )


if __name__ == "__main__":
    main()
