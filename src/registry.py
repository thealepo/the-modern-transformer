VARIANTS: dict[str , tuple[str , ...]] = {
    "norm": ("layernorm",),
    "positional": ("learned",),
    "ffn": ("gelu_mlp",),
    "attention": ("mha",),
}


def resolve(slot: str , name: str):
    # TODO(promote)
    import model

    tables = {
        "norm": {"layernorm": model.LayerNorm},
        "positional": {"learned": model.LearnedPositional},
        "ffn": {"gelu_mlp": model.MultiLayerPerceptron},
        "attention": {"mha": model.MultiHeadAttention},
    }
    return tables[slot][name]
