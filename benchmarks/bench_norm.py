from harness import main_cli

CONFIGS = ['configs/baseline.yaml' , 'configs/rmsnorm.yaml']

if __name__ == "__main__":
    main_cli('norm' , CONFIGS)
