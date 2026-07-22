from dataclasses import dataclass

_VALID_VARIANTS = {
    "norm": ("layernorm",),
    "positional": ("learned",),
    "ffn": ("gelu_mlp",),
    "attention": ("mha",),
}


@dataclass(frozen=True , kw_only=True , slots=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int
    hidden_size: int
    n_heads: int
    n_layers: int

    norm: str
    positional: str
    ffn: str
    attention: str

    n_kv_heads: int | None = None
    mlp_ratio: int = 4

    def __post_init__(self):
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not divisible by "
                f"n_heads={self.n_heads}"
            )
        for slot, valid in _VALID_VARIANTS.items():
            name = getattr(self, slot)
            if name not in valid:
                raise ValueError(
                    f"unknown {slot}={name!r}; valid options: {', '.join(valid)}"
                )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.mlp_ratio * self.hidden_size

    @classmethod
    def from_yaml(cls , path) -> "TransformerConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
