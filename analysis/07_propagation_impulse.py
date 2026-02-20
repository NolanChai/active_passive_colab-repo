from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from analysis.common import (
    METRIC_LABELS,
    METRICS,
    bootstrap_ci,
    build_pair_table,
    ensure_dir,
)
from src.uid import run_uid_pipeline

TARGET_METRICS = METRICS + ["uid_slope_abs"]


def metric_col(metric: str) -> str:
    return "d_norm_uid_slope_abs" if metric == "uid_slope_abs" else f"d_norm_{metric}"


def compute_impulse(pair_df: pd.DataFrame, k_max: int) -> pd.DataFrame:
    rows = []
    sub = pair_df[(pair_df["target_offset"] >= 0) & (pair_df["target_offset"] <= k_max)].copy()

    for context in sorted(sub["context"].dropna().unique()):
        ctx_df = sub[sub["context"] == context]
        for direction in ["A to P", "P to A"]:
            dir_df = ctx_df[ctx_df["direction"] == direction]
            if dir_df.empty:
                continue

            for metric in TARGET_METRICS:
                col = metric_col(metric)
                means = {}
                for k in range(k_max + 1):
                    vals = dir_df[dir_df["target_offset"] == k][col].dropna().to_numpy(dtype=float)
                    if vals.size == 0:
                        continue
                    means[k] = float(np.mean(vals))
                    ci_lo, ci_hi = bootstrap_ci(vals, np.mean)
                    rows.append(
                        {
                            "row_type": "offset",
                            "context": context,
                            "direction": direction,
                            "metric": metric,
                            "metric_label": METRIC_LABELS[metric],
                            "offset": k,
                            "n": int(vals.size),
                            "mean_norm_delta": means[k],
                            "median_norm_delta": float(np.median(vals)),
                            "ci_low": ci_lo,
                            "ci_high": ci_hi,
                            "positive_rate": float((vals > 0).mean()),
                        }
                    )

                if 0 not in means:
                    continue
                k0 = means[0]
                mag0 = abs(k0)
                thresh = 0.05 * mag0
                half_life = np.nan
                if mag0 > 0:
                    for k in sorted(means.keys()):
                        if k == 0:
                            continue
                        if abs(means[k]) <= thresh:
                            half_life = float(k)
                            break

                ks = np.array(sorted(means.keys()), dtype=float)
                ys = np.array([abs(means[int(k)]) + 1e-8 for k in ks], dtype=float)
                decay = np.nan
                if ks.size > 1:
                    decay = float(np.polyfit(ks, np.log(ys), 1)[0])

                rows.append(
                    {
                        "row_type": "decay",
                        "context": context,
                        "direction": direction,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "offset": -1,
                        "n": int(len(ks)),
                        "mean_norm_delta": float(k0),
                        "median_norm_delta": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "positive_rate": np.nan,
                        "immediate_k0": float(k0),
                        "decay_log_slope": decay,
                        "half_life_k": half_life,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagation/impulse analysis after active-passive substitution.")
    parser.add_argument("--conllu", type=Path, default=Path("data/en_gum-ud-train.conllu"))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--contexts", type=str, default="document")
    parser.add_argument("--uid_level", type=str, default="sentence")
    parser.add_argument("--uid_unit", type=str, default="word")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit_docs", type=int, default=None)
    parser.add_argument("--limit_sents_per_doc", type=int, default=None)
    parser.add_argument("--raw_output", type=Path, default=Path("analysis/results/uid_post_word_sentence.csv"))
    parser.add_argument("--summary_output", type=Path, default=Path("analysis/results/impulse_response_summary.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.raw_output.parent)

    if args.raw_output.exists() and not args.force:
        uid_df = pd.read_csv(args.raw_output)
    else:
        contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
        uid_df = run_uid_pipeline(
            args.conllu,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            context_levels=contexts,
            generate_counterfactual=True,
            uid_level=args.uid_level,
            uid_unit=args.uid_unit,
            device=args.device,
            output_dir=args.raw_output.parent,
            output_file=Path(args.raw_output.name),
            include_raw_surps=False,
            eval_scope="post",
            verbose=False,
        )
        uid_df.to_csv(args.raw_output, index=False)

    pair_df = build_pair_table(uid_df)
    impulse_df = compute_impulse(pair_df, k_max=args.k)
    impulse_df.to_csv(args.summary_output, index=False)

    print(f"Wrote raw post data: {args.raw_output}")
    print(f"Wrote summary: {args.summary_output}")


if __name__ == "__main__":
    main()
