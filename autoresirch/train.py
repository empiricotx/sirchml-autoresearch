from __future__ import annotations

import torch
from torch import nn

from autoresirch.prepare import ArchitectureContext, ArchitectureSpec, run_experiment


ARCHITECTURE = ArchitectureSpec(
    family="mlp",
    hidden_dims=(48,),
    activation="relu",
    dropout=0.0,
    normalization="none",
    use_bias=True,
    use_rnafm_embeddings=False,
)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _normalization(name: str, width: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "layernorm":
        return nn.LayerNorm(width)
    if name == "batchnorm":
        return nn.BatchNorm1d(width)
    raise ValueError(f"Unsupported normalization: {name}")


class SimpleMLP(nn.Module):
    def __init__(self, context: ArchitectureContext, architecture: ArchitectureSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = context.input_dim

        for hidden_dim in architecture.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=architecture.use_bias))
            layers.append(_normalization(architecture.normalization, hidden_dim))
            layers.append(_activation(architecture.activation))
            if architecture.dropout > 0:
                layers.append(nn.Dropout(architecture.dropout))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(input_dim, context.output_dim, bias=architecture.use_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))


def build_model(context: ArchitectureContext) -> nn.Module:
    return SimpleMLP(context, ARCHITECTURE)


def main() -> None:
    run_experiment(ARCHITECTURE, build_model)


if __name__ == "__main__":
    main()
