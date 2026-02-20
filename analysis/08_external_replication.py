from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from analysis.common import METRIC_LABELS, METRICS, PRIMARY_CONTEXTS, build_pair_table, ensure_dir
from src.uid import run_uid_pipeline

TARGET_METRICS = METRICS + ["uid_slope_abs"]


def metric_col(metric: str) -> str:
    return "d_norm_uid_slope_abs" if metric == "uid_slope_abs" else f"d_norm_{metric}"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in PRIMARY_CONTEXTS:
        sub = df[df["context"] == context]
        if sub.empty:
            continue
        for metric in TARGET_METRICS:
            col = metric_col(metric)
            vals = sub[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "context": context,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": int(vals.size),
                    "mean_norm_delta": float(np.mean(vals)),
                    "median_norm_delta": float(np.median(vals)),
                    "positive_rate": float((vals > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 out-of-corpus replication runner.")
    parser.add_argument("--conllu", type=Path, required=True, help="Path to external UD .conllu file")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--uid_level", type=str, default="sentence")
    parser.add_argument("--uid_unit", type=str, default="word")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit_docs", type=int, default=None)
    parser.add_argument("--limit_sents_per_doc", type=int, default=None)
    parser.add_argument("--output_raw", type=Path, default=Path("analysis/results/external_uid_raw.csv"))
    parser.add_argument("--output_summary", type=Path, default=Path("analysis/results/external_replication_summary.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_raw.parent)

    if args.output_raw.exists() and not args.force:
        uid_df = pd.read_csv(args.output_raw)
    else:
        uid_df = run_uid_pipeline(
            args.conllu,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            context_levels=PRIMARY_CONTEXTS,
            generate_counterfactual=True,
            uid_level=args.uid_level,
            uid_unit=args.uid_unit,
            device=args.device,
            output_dir=args.output_raw.parent,
            output_file=Path(args.output_raw.name),
            include_raw_surps=False,
            eval_scope="target",
            verbose=False,
        )
        uid_df.to_csv(args.output_raw, index=False)

    pair_df = build_pair_table(uid_df)
    changed = pair_df[pair_df["is_target_sentence"] & pair_df["changed"]].copy()
    summary = summarize(changed)
    summary.to_csv(args.output_summary, index=False)

    print(f"Wrote: {args.output_raw}")
    print(f"Wrote: {args.output_summary}")


if __name__ == "__main__":
    main()
