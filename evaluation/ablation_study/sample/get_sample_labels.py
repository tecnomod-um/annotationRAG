"""Draw a stratified random sample of labels from a pool TSV.

The pool must already be disjoint from the internal evaluation set (e.g. the 4,442
labels not used in the evaluation). Passing --exclude (optional) adds a safety check
that no sampled label belongs to the internal evaluation set.

Usage:
    python sample_external_labels.py --pool pool.tsv --out external_sample_100.tsv --seed 42
    python sample_external_labels.py --pool pool.tsv --exclude mappings_test.tsv ...
"""
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--pool", required=True, help="TSV to sample from (already disjoint from the evaluation set)")
p.add_argument("--exclude", default=None, help="optional TSV with internal evaluation labels (safety check)")
p.add_argument("--out", default="external_sample_100.tsv")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--label-col", default=None, help="label column (default: first column)")
p.add_argument("--type-col", default="Type", help="concept type column (CL / CT / A)")
p.add_argument("--n-cl", type=int, default=50)
p.add_argument("--n-ct", type=int, default=38)
p.add_argument("--n-a", type=int, default=12)
args = p.parse_args()

read = lambda f: pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
pool = read(args.pool)

label_col = args.label_col or pool.columns[0]
norm = lambda s: s.str.strip()

internal_labels = set()
if args.exclude:
    excl = read(args.exclude)
    internal_labels = set(norm(excl[args.label_col or excl.columns[0]]))

n_before = len(pool)
pool = pool[~norm(pool[label_col]).isin(internal_labels)]
pool = pool[pool[args.type_col].isin(["CL", "CT", "A"])]
pool = pool.drop_duplicates(subset=label_col)
print(f"Pool file: {n_before} rows | usable typed labels not in --exclude: {len(pool)}")

quota = {"CL": args.n_cl, "CT": args.n_ct, "A": args.n_a}
parts = []
for t, n in quota.items():
    candidates = pool[pool[args.type_col] == t]
    if len(candidates) < n:
        raise SystemExit(f"Not enough {t} labels in the pool: {len(candidates)} < {n}")
    parts.append(candidates.sample(n, random_state=args.seed))
sample = pd.concat(parts).reset_index(drop=True)

# Sanity checks
assert len(sample) == sum(quota.values())
assert sample[label_col].is_unique
assert not set(norm(sample[label_col])) & internal_labels, "overlap with the excluded evaluation set!"

sample.to_csv(args.out, sep="\t", index=False)
print(sample[args.type_col].value_counts().to_dict(), f"| seed={args.seed} | saved to {args.out}")