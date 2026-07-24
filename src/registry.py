VARIANTS: dict[str , tuple[str , ...]] = {
    'norm': ('layernorm' , 'rmsnorm'),
    'positional': ('learned',),
    'ffn': ('gelu_mlp' , 'swiglu_mlp'),
    'attention': ('mha',),
}


def resolve(slot: str , name: str):
    # TODO(promote)
    import model

    tables = {
        'norm': {'layernorm': model.LayerNorm , 'rmsnorm': model.RMSNorm},
        'positional': {'learned': model.LearnedPositional},
        'ffn': {'gelu_mlp': model.MultiLayerPerceptron , 'swiglu_mlp': model.SwiGLU},
        'attention': {'mha': model.MultiHeadAttention},
    }
    return tables[slot][name]
