from harness import main_cli

CONFIGS = ['configs/baseline.yaml' , 'configs/swiglu.yaml']

if __name__ == "__main__":
    main_cli('ffn' , CONFIGS)
