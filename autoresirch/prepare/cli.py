from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from autoresirch.prepare import DATASET_CONFIG, ExperimentMode
from autoresirch.prepare.standard.dataset import prepare_dataset, print_dataset_summary

EXPERIMENT_MODE_CHOICES: tuple[ExperimentMode, ...] = ("standard", "comparative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed siRNA regression dataset and experiment harness."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached prepared dataset even if the cache looks current.",
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=None,
        help="Override DatasetConfig.raw_data_path for this preparation run.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODE_CHOICES,
        default=DATASET_CONFIG.experiment_mode,
        help="Override DatasetConfig.experiment_mode for this preparation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DATASET_CONFIG
    dataset_config = replace(DATASET_CONFIG, experiment_mode=args.experiment_mode)
    if args.raw_data_path is not None:
        dataset_config = replace(dataset_config, raw_data_path=args.raw_data_path)
    prepared = prepare_dataset(dataset_config=dataset_config, force=args.force)
    print_dataset_summary(prepared)


if __name__ == "__main__":
    main()
