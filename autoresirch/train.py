from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from autoresirch.prepare import ArchitectureContext, ArchitectureSpec, run_experiment


ARCHITECTURE = ArchitectureSpec(
    family="hybrid_cnn_mlp",
    hidden_dims=(64,),
    activation="silu",
    dropout=0.1,
    normalization="none",
    use_bias=True,
    use_rnafm_embeddings=True,
    sequence_feature_source="rnafm::antisense_strand_seq",
    conv_channels=(32, 64),
    kernel_sizes=(5, 3),
    pooling="mean",
    flat_hidden_dims=(192, 96),
    fusion_hidden_dims=(160, 64),
    rnafm_pooling_strategy="mean",
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


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str,
        normalization: str,
        dropout: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.normalization = make_normalization(normalization, output_dim)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class SequenceConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        activation: str,
        dropout: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=use_bias,
        )
        self.normalization = nn.BatchNorm1d(output_channels)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class FlatEncoder(nn.Module):
    def __init__(self, input_dim: int, architecture: ArchitectureSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_width = input_dim
        for width in architecture.flat_hidden_dims:
            layers.append(
                MLPBlock(
                    previous_width,
                    width,
                    activation=architecture.activation,
                    normalization=architecture.normalization,
                    dropout=architecture.dropout,
                    use_bias=architecture.use_bias,
                )
            )
            previous_width = width
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = previous_width

    def forward(self, flat: torch.Tensor | None) -> torch.Tensor | None:
        if flat is None:
            return None
        return self.encoder(flat)


class SequenceEncoder(nn.Module):
    def __init__(self, embedding_dim: int, architecture: ArchitectureSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_channels = embedding_dim
        for channels, kernel_size in zip(
            architecture.conv_channels, architecture.kernel_sizes, strict=True
        ):
            layers.append(
                SequenceConvBlock(
                    previous_channels,
                    channels,
                    kernel_size=kernel_size,
                    activation=architecture.activation,
                    dropout=architecture.dropout,
                    use_bias=architecture.use_bias,
                )
            )
            previous_channels = channels
        self.encoder = nn.Sequential(*layers)
        self.output_dim = previous_channels
        self.pooling = architecture.pooling

    def forward(self, sequence: torch.Tensor | None) -> torch.Tensor | None:
        if sequence is None:
            return None
        outputs = sequence.transpose(1, 2)
        outputs = self.encoder(outputs)
        if self.pooling == "max":
            outputs = F.adaptive_max_pool1d(outputs, 1)
        else:
            outputs = F.adaptive_avg_pool1d(outputs, 1)
        return outputs.squeeze(-1)


class HybridRegressor(nn.Module):
    def __init__(self, context: ArchitectureContext, architecture: ArchitectureSpec) -> None:
        super().__init__()
        flat_input_dim = context.flat_input_dim or 0
        sequence_embedding_dim = context.sequence_embedding_dim or 0
        self.flat_encoder = FlatEncoder(flat_input_dim, architecture)
        self.sequence_encoder = SequenceEncoder(sequence_embedding_dim, architecture)
        fusion_input_dim = self.flat_encoder.output_dim + self.sequence_encoder.output_dim

        fusion_layers: list[nn.Module] = []
        previous_width = fusion_input_dim
        for width in architecture.fusion_hidden_dims:
            fusion_layers.append(
                MLPBlock(
                    previous_width,
                    width,
                    activation=architecture.activation,
                    normalization=architecture.normalization,
                    dropout=architecture.dropout,
                    use_bias=architecture.use_bias,
                )
            )
            previous_width = width
        self.fusion = nn.Sequential(*fusion_layers)
        self.head = nn.Linear(previous_width, context.output_dim, bias=architecture.use_bias)

    def forward(
        self,
        *,
        flat: torch.Tensor | None = None,
        sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_features = self.flat_encoder(flat)
        sequence_features = self.sequence_encoder(sequence)
        if flat_features is None or sequence_features is None:
            raise ValueError("HybridRegressor requires both flat and sequence inputs.")
        fused = torch.cat((flat_features, sequence_features), dim=1)
        hidden = self.fusion(fused)
        return self.head(hidden)


def build_model(context: ArchitectureContext) -> nn.Module:
    return HybridRegressor(context, ARCHITECTURE)


def main() -> None:
    run_experiment(ARCHITECTURE, build_model)


if __name__ == "__main__":
    main()
