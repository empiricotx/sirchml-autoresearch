from __future__ import annotations

import torch
from torch import nn

from autoresirch.prepare import ArchitectureContext, ArchitectureSpec, run_experiment


ARCHITECTURE = ArchitectureSpec(
    family="hybrid_cnn_mlp",
    hidden_dims=(),
    activation="silu",
    dropout=0.1,
    normalization="none",
    use_bias=True,
    use_rnafm_embeddings=True,
    sequence_feature_source="rnafm::antisense_strand_seq",
    conv_channels=(64, 128),
    kernel_sizes=(5, 3),
    pooling="mean",
    flat_hidden_dims=(256, 128),
    fusion_hidden_dims=(128, 64),
    rnafm_embedding_dim=16,
)

USE_INPUT_SKIP = False


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
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: ArchitectureSpec,
        *,
        hidden_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_width = input_dim
        residual = architecture.family == "residual_mlp"
        widths = architecture.hidden_dims if hidden_dims is None else hidden_dims
        for width in widths:
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
        self.output_dim = previous_width
        self.head = nn.Linear(previous_width, output_dim, bias=architecture.use_bias)
        self.input_skip = (
            nn.Linear(input_dim, output_dim, bias=False) if USE_INPUT_SKIP else None
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def forward(
        self,
        flat: torch.Tensor | None = None,
        *,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flat is None:
            raise ValueError("RegressionMLP requires flat features.")
        hidden = self.encode(flat)
        outputs = self.head(hidden)
        if self.input_skip is not None:
            outputs = outputs + self.input_skip(flat)
        return outputs


class SequenceConvEncoder(nn.Module):
    def __init__(self, context: ArchitectureContext, architecture: ArchitectureSpec) -> None:
        super().__init__()
        if context.sequence_embedding_dim is None or context.sequence_length is None:
            raise ValueError("SequenceConvEncoder requires sequence metadata in the context.")
        layers: list[nn.Module] = []
        in_channels = context.sequence_embedding_dim
        for out_channels, kernel_size in zip(
            architecture.conv_channels,
            architecture.kernel_sizes,
            strict=True,
        ):
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=architecture.use_bias,
                    ),
                    nn.BatchNorm1d(out_channels),
                    make_activation(architecture.activation),
                    nn.Dropout(architecture.dropout),
                ]
            )
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        self.output_dim = in_channels
        if architecture.pooling == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone(sequence.transpose(1, 2))
        pooled = self.pool(encoded).squeeze(-1)
        return pooled


class ConvRegressionModel(nn.Module):
    def __init__(self, context: ArchitectureContext, architecture: ArchitectureSpec) -> None:
        super().__init__()
        self.encoder = SequenceConvEncoder(context, architecture)
        self.head = nn.Linear(self.encoder.output_dim, context.output_dim, bias=architecture.use_bias)

    def forward(
        self,
        flat: torch.Tensor | None = None,
        *,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del flat
        if sequence is None:
            raise ValueError("ConvRegressionModel requires sequence features.")
        hidden = self.encoder(sequence)
        return self.head(hidden)


class HybridRegressionModel(nn.Module):
    def __init__(self, context: ArchitectureContext, architecture: ArchitectureSpec) -> None:
        super().__init__()
        if context.flat_input_dim is None:
            raise ValueError("HybridRegressionModel requires flat features.")
        self.sequence_encoder = SequenceConvEncoder(context, architecture)
        self.flat_encoder = RegressionMLP(
            input_dim=context.flat_input_dim,
            output_dim=context.output_dim,
            architecture=architecture,
            hidden_dims=architecture.flat_hidden_dims,
        )

        fusion_layers: list[nn.Module] = []
        previous_width = self.sequence_encoder.output_dim + self.flat_encoder.output_dim
        for width in architecture.fusion_hidden_dims:
            fusion_layers.append(
                FeedForwardBlock(
                    previous_width,
                    width,
                    activation=architecture.activation,
                    normalization=architecture.normalization,
                    dropout=architecture.dropout,
                    use_bias=architecture.use_bias,
                    residual=False,
                )
            )
            previous_width = width
        self.fusion = nn.Sequential(*fusion_layers) if fusion_layers else nn.Identity()
        self.head = nn.Linear(previous_width, context.output_dim, bias=architecture.use_bias)

    def forward(
        self,
        flat: torch.Tensor | None = None,
        *,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flat is None or sequence is None:
            raise ValueError("HybridRegressionModel requires both flat and sequence features.")
        flat_hidden = self.flat_encoder.encode(flat)
        sequence_hidden = self.sequence_encoder(sequence)
        fused = torch.cat([flat_hidden, sequence_hidden], dim=1)
        return self.head(self.fusion(fused))


def build_model(context: ArchitectureContext) -> nn.Module:
    if ARCHITECTURE.family in {"mlp", "residual_mlp"}:
        return RegressionMLP(
            input_dim=context.input_dim,
            output_dim=context.output_dim,
            architecture=ARCHITECTURE,
        )
    if ARCHITECTURE.family == "cnn":
        return ConvRegressionModel(context, ARCHITECTURE)
    if ARCHITECTURE.family == "hybrid_cnn_mlp":
        return HybridRegressionModel(context, ARCHITECTURE)
    raise ValueError(f"Unsupported architecture family: {ARCHITECTURE.family!r}")


def main() -> None:
    run_experiment(ARCHITECTURE, build_model)


if __name__ == "__main__":
    main()
