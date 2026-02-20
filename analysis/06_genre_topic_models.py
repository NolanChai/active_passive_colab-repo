from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from analysis.common import METRIC_LABELS, METRICS, ensure_dir, p_adjust_bh, p_adjust_holm

TARGET_METRICS = METRICS + ["uid_slope_abs"]


def metric_col(metric: str) -> str:
    return "d_norm_uid_slope_abs" if metric == "uid_slope_abs" else f"d_norm_{metric}"


def effect_by_group(df: pd.DataFrame, group_col: str, min_n: int = 20) -> pd.DataFrame:
    rows = []
    grouped_values = df[group_col].value_counts()
    keep = set(grouped_values[grouped_values >= min_n].index)
    work = df.copy()
    work[f"{group_col}_group"] = work[group_col].where(work[group_col].isin(keep), "OTHER")

    for metric in TARGET_METRICS:
        col = metric_col(metric)
        for grp, g in work.groupby(f"{group_col}_group"):
            vals = g[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "row_type": "effect_size",
                    "grouping": group_col,
                    "group": grp,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": int(vals.size),
                    "mean_norm_delta": float(np.mean(vals)),
                    "median_norm_delta": float(np.median(vals)),
                    "positive_rate": float((vals > 0).mean()),
                }
            )
    return pd.DataFrame(rows), work


def interaction_tests(df: pd.DataFrame, group_col: str, grouped_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return pd.DataFrame()

    rows = []
    gcol = f"{group_col}_group"

    for metric in TARGET_METRICS:
        col = metric_col(metric)
        mdf = grouped_df[[col, "direction", "pair_id", gcol]].dropna().rename(columns={col: "y"})
        if mdf.empty or mdf["direction"].nunique() < 2 or mdf[gcol].nunique() < 2:
            continue

        fit = smf.ols(f"y ~ C(direction) * C({gcol})", data=mdf).fit(
            cov_type="cluster", cov_kwds={"groups": mdf["pair_id"]}
        )
        for term, coef, pval, se, tval in zip(fit.params.index, fit.params.values, fit.pvalues, fit.bse, fit.tvalues):
            if "C(direction)" not in term:
                continue
            rows.append(
                {
                    "row_type": "interaction",
                    "grouping": group_col,
                    "group": term,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": int(mdf.shape[0]),
                    "mean_norm_delta": np.nan,
                    "median_norm_delta": np.nan,
                    "positive_rate": np.nan,
                    "coef": float(coef),
                    "se": float(se),
                    "t_or_z": float(tval),
                    "pvalue": float(pval),
                    "r2": float(fit.rsquared),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pvalue_fdr_bh"] = p_adjust_bh(out["pvalue"].to_numpy())
        out["pvalue_holm"] = p_adjust_holm(out["pvalue"].to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Genre/topic heterogeneity for UID effects.")
    parser.add_argument("--pairs", type=Path, default=Path("analysis/results/pairwise_changed.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/results/genre_topic_heterogeneity.csv"))
    args = parser.parse_args()

    ensure_dir(args.output.parent)

    df = pd.read_csv(args.pairs)
    doc = df[df["context"] == "document"].copy()

    genre_effect, genre_grouped = effect_by_group(doc, "genre", min_n=20)
    topic_effect, topic_grouped = effect_by_group(doc, "topic_slug", min_n=20)

    genre_inter = interaction_tests(doc, "genre", genre_grouped)
    topic_inter = interaction_tests(doc, "topic_slug", topic_grouped)

    out = pd.concat([genre_effect, topic_effect, genre_inter, topic_inter], ignore_index=True, sort=False)
    out.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
