VARIANTS: dict[str , tuple[str , ...]] = {
    'norm': ('layernorm' , 'rmsnorm'),
    'positional': ('learned',),
    'ffn': ('gelu_mlp' , 'swiglu_mlp'),
    'attention': ('mha',),
}


def resolve(slot: str , name: str):
    # TODO(promote)
    from layers import norm , ffn
    import model

    tables = {
        'norm': {'layernorm': norm.LayerNorm , 'rmsnorm': norm.RMSNorm},
        'positional': {'learned': model.LearnedPositional},
        'ffn': {'gelu_mlp': ffn.MultiLayerPerceptron , 'swiglu_mlp': ffn.SwiGLU},
        'attention': {'mha': model.MultiHeadAttention},
    }
    return tables[slot][name]
