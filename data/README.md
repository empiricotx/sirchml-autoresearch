# Dataset Schema

Place the raw siRNA regression dataset in this directory.

Default location:

- `feature_subset_df.csv`

That resolves to [data/feature_subset_df.csv](/Users/lucasplatter/sirchml-autoresearch/data/feature_subset_df.csv).

Default required columns:

- `transcript_gene`: gene identifier used for held-out folds
- `rel_exp_individual`: regression target
- `antisense_strand_seq`: row identifier, excluded from features

Optional columns:

- additional numeric covariates
- categorical metadata
- additional sequence columns

By default, all other columns are treated as feature candidates. If your file path or column names differ, update `DATASET_CONFIG` in [prepare.py](/Users/lucasplatter/sirchml-autoresearch/prepare.py).
