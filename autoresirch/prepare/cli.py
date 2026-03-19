from __future__ import annotations

import argparse

from autoresirch.prepare.standard.dataset import prepare_dataset, print_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed siRNA regression dataset and experiment harness."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached prepared dataset even if the cache looks current.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_dataset(force=args.force)
    print_dataset_summary(prepared)


if __name__ == "__main__":
    main()
