from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _normalize_sequence(value: object, allowed_bases: set[str], unknown_base: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    sequence = str(value).strip().upper().replace("T", "U")
    return "".join(char if char in allowed_bases else unknown_base for char in sequence)


def _build_base_embedding_table(embedding_dim: int) -> dict[str, np.ndarray]:
    base_vectors = {
        "A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "C": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "G": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "U": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "N": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    }
    if embedding_dim <= 4:
        return {base: vector[:embedding_dim].copy() for base, vector in base_vectors.items()}

    table: dict[str, np.ndarray] = {}
    for base, vector in base_vectors.items():
        padded = np.zeros(embedding_dim, dtype=np.float32)
        padded[:4] = vector
        for offset in range(4, embedding_dim):
            padded[offset] = ((offset + 1) * (ord(base) - 63)) % 17 / 17.0
        table[base] = padded
    return table


def build_rnafm_embedding_tensor(
    sequences: pd.Series,
    *,
    max_length: int,
    embedding_dim: int,
    allowed_bases: str,
    unknown_base: str,
) -> np.ndarray:
    """Build a deterministic embedding tensor shaped [rows, length, embedding_dim]."""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")

    allowed_set = set(allowed_bases)
    base_embeddings = _build_base_embedding_table(embedding_dim)
    normalized = [
        _normalize_sequence(value, allowed_set, unknown_base)
        for value in sequences.fillna("")
    ]

    tensor = np.zeros((len(normalized), max_length, embedding_dim), dtype=np.float32)
    for row_index, sequence in enumerate(normalized):
        for position in range(min(len(sequence), max_length)):
            base = sequence[position]
            tensor[row_index, position] = base_embeddings.get(base, base_embeddings[unknown_base])
            if embedding_dim > 4:
                position_scale = (position + 1) / max_length
                tensor[row_index, position, 4:] += position_scale
    return tensor
