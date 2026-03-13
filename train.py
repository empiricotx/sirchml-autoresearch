from __future__ import annotations

import torch
from torch import nn

from prepare import ArchitectureContext, ArchitectureSpec, run_experiment


ARCHITECTURE = ArchitectureSpec(
    family="mlp",
    hidden_dims=(320, 160, 80, 40),
    activation="silu",
    dropout=0.02,
    normalization="none",
    use_bias=False,
)


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def make_normalization(name: str, width: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "layernorm":
        return nn.LayerNorm(width)
    if name == "batchnorm":
        return nn.BatchNorm1d(width)
    raise ValueError(f"Unsupported normalization: {name}")


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str,
        normalization: str,
        dropout: float,
        use_bias: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.normalization = make_normalization(normalization, output_dim)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = residual
        if residual and input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + self.skip(inputs)
        return outputs


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, architecture: ArchitectureSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_width = input_dim
        residual = architecture.family == "residual_mlp"
        for width in architecture.hidden_dims:
            layers.append(
                FeedForwardBlock(
                    previous_width,
                    width,
                    activation=architecture.activation,
                    normalization=architecture.normalization,
                    dropout=architecture.dropout,
                    use_bias=architecture.use_bias,
                    residual=residual,
                )
            )
            previous_width = width
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(previous_width, output_dim, bias=architecture.use_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        return self.head(hidden)


def build_model(context: ArchitectureContext) -> nn.Module:
    return RegressionMLP(
        input_dim=context.input_dim,
        output_dim=context.output_dim,
        architecture=ARCHITECTURE,
    )


def main() -> None:
    run_experiment(ARCHITECTURE, build_model)


if __name__ == "__main__":
    main()
