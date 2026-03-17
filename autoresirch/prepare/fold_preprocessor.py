from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .schemas import PreparedDataset, FoldSpec


@dataclass
class FoldPreprocessor:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    numeric_fill_values: dict[str, float] | None = None
    numeric_means: dict[str, float] | None = None
    numeric_stds: dict[str, float] | None = None
    categorical_levels: dict[str, tuple[str, ...]] | None = None
    feature_names: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        self.numeric_fill_values = {}
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_levels = {}
        feature_names: list[str] = []

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if not values.dropna().empty else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std <= 0:
                std = 1.0
            self.numeric_fill_values[column] = median
            self.numeric_means[column] = mean
            self.numeric_stds[column] = std
            feature_names.append(column)

        for column in self.categorical_columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            levels = tuple(sorted(values.unique()))
            if "__UNK__" not in levels:
                levels = levels + ("__UNK__",)
            self.categorical_levels[column] = levels
            feature_names.extend(f"{column}=={level}" for level in levels)

        self.feature_names = tuple(feature_names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if (
            self.numeric_fill_values is None
            or self.numeric_means is None
            or self.numeric_stds is None
            or self.categorical_levels is None
        ):
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        arrays: list[np.ndarray] = []

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            filled = values.fillna(self.numeric_fill_values[column]).to_numpy(dtype=np.float32)
            standardized = (filled - self.numeric_means[column]) / self.numeric_stds[column]
            arrays.append(standardized[:, None].astype(np.float32))

        for column in self.categorical_columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            levels = self.categorical_levels[column]
            known_levels = set(levels)
            normalized = values.where(values.isin(known_levels), "__UNK__")
            encoded = np.stack(
                [
                    (normalized == level).to_numpy(dtype=np.float32)
                    for level in levels
                ],
                axis=1,
            )
            arrays.append(encoded.astype(np.float32))

        if not arrays:
            raise ValueError("FoldPreprocessor produced an empty feature matrix.")

        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, target: np.ndarray) -> "TargetScaler":
        mean = float(target.mean())
        std = float(target.std())
        if std <= 0:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, target: np.ndarray) -> np.ndarray:
        return ((target - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, target: np.ndarray) -> np.ndarray:
        return (target * self.std + self.mean).astype(np.float32)


def build_cv_folds(prepared: PreparedDataset) -> list[FoldSpec]:
    train_gene_set = set(prepared.train_genes)
    folds: list[FoldSpec] = []
    for gene in prepared.cv_genes:
        val_mask = prepared.genes == gene
        train_mask = np.isin(prepared.genes, tuple(train_gene_set - {gene}))
        val_indices = np.flatnonzero(val_mask)
        train_indices = np.flatnonzero(train_mask)
        if val_indices.size == 0 or train_indices.size == 0:
            raise ValueError(f"Fold for gene {gene!r} has empty train or validation rows.")
        folds.append(FoldSpec(gene=gene, train_indices=train_indices, val_indices=val_indices))
    return folds