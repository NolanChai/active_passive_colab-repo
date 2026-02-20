from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.common import (
    ALL_CONTEXTS,
    METRIC_LABELS,
    METRICS,
    bootstrap_ci,
    cluster_robust_ols,
    cohen_dz,
    ensure_dir,
    p_adjust_bh,
    p_adjust_holm,
    signflip_test,
    wilcoxon_p,
)

TARGET_METRICS = METRICS + ["uid_slope_abs"]


def metric_col(metric: str) -> str:
    return "d_norm_uid_slope_abs" if metric == "uid_slope_abs" else f"d_norm_{metric}"


def summarize(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for context in ALL_CONTEXTS:
        sub_ctx = df[df["context"] == context]
        if sub_ctx.empty:
            continue
        for metric in TARGET_METRICS:
            col = metric_col(metric)
            vals = sub_ctx[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mean_lo, mean_hi = bootstrap_ci(vals, np.mean)
            med_lo, med_hi = bootstrap_ci(vals, np.median)
            obs, perm_two, perm_one = signflip_test(vals)
            wil_two, wil_one = wilcoxon_p(vals, alternative="greater")
            rows.append(
                {
                    "subset": subset,
                    "context": context,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": vals.size,
                    "mean_norm_delta": float(np.mean(vals)),
                    "median_norm_delta": float(np.median(vals)),
                    "mean_ci_low": mean_lo,
                    "mean_ci_high": mean_hi,
                    "median_ci_low": med_lo,
                    "median_ci_high": med_hi,
                    "cohen_dz": cohen_dz(vals),
                    "sign_positive_rate": float((vals > 0).mean()),
                    "perm_obs_mean": obs,
                    "perm_p_two": perm_two,
                    "perm_p_one_factual_better": perm_one,
                    "wilcoxon_p_two": wil_two,
                    "wilcoxon_p_one_factual_better": wil_one,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["perm_p_two_fdr_bh"] = p_adjust_bh(out["perm_p_two"].to_numpy())
    out["perm_p_two_holm"] = p_adjust_holm(out["perm_p_two"].to_numpy())
    out["wilcoxon_p_two_fdr_bh"] = p_adjust_bh(out["wilcoxon_p_two"].to_numpy())
    out["wilcoxon_p_two_holm"] = p_adjust_holm(out["wilcoxon_p_two"].to_numpy())
    return out


def summarize_by_direction(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for direction in ["A to P", "P to A"]:
        sub_dir = df[df["direction"] == direction]
        if sub_dir.empty:
            continue
        summary = summarize(sub_dir, subset=subset)
        if summary.empty:
            continue
        summary["direction"] = direction
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)

    adjusted = []
    for direction, g in out.groupby("direction", sort=False):
        g = g.copy()
        g["perm_p_two_fdr_bh_by_dir"] = p_adjust_bh(g["perm_p_two"].to_numpy())
        g["perm_p_two_holm_by_dir"] = p_adjust_holm(g["perm_p_two"].to_numpy())
        g["wilcoxon_p_two_fdr_bh_by_dir"] = p_adjust_bh(g["wilcoxon_p_two"].to_numpy())
        g["wilcoxon_p_two_holm_by_dir"] = p_adjust_holm(g["wilcoxon_p_two"].to_numpy())
        adjusted.append(g)
    return pd.concat(adjusted, ignore_index=True)


def balanced_direction_downsample(df: pd.DataFrame, n_resamples: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for context in ALL_CONTEXTS:
        sub = df[df["context"] == context]
        a2p = sub[sub["direction"] == "A to P"]
        p2a = sub[sub["direction"] == "P to A"]
        if a2p.empty or p2a.empty:
            continue

        for metric in TARGET_METRICS:
            col = metric_col(metric)
            vals_a = a2p[col].dropna().to_numpy(dtype=float)
            vals_p = p2a[col].dropna().to_numpy(dtype=float)
            n = min(vals_a.size, vals_p.size)
            if n < 8:
                continue

            combined_means = []
            gaps = []
            for _ in range(n_resamples):
                sa = vals_a[rng.choice(vals_a.size, size=n, replace=False)]
                sp = vals_p[rng.choice(vals_p.size, size=n, replace=False)]
                combined_means.append(float(np.mean(np.concatenate([sa, sp]))))
                gaps.append(float(np.mean(sp) - np.mean(sa)))

            combined_means = np.asarray(combined_means)
            gaps = np.asarray(gaps)
            rows.append(
                {
                    "test_type": "balanced_downsample",
                    "context": context,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n_small": n,
                    "n_a2p": vals_a.size,
                    "n_p2a": vals_p.size,
                    "balanced_mean": float(np.mean(combined_means)),
                    "balanced_ci_low": float(np.quantile(combined_means, 0.025)),
                    "balanced_ci_high": float(np.quantile(combined_means, 0.975)),
                    "direction_gap_p2a_minus_a2p": float(np.mean(gaps)),
                    "gap_ci_low": float(np.quantile(gaps, 0.025)),
                    "gap_ci_high": float(np.quantile(gaps, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def direction_context_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["context"] = pd.Categorical(work["context"], categories=ALL_CONTEXTS, ordered=True)

    for metric in TARGET_METRICS:
        col = metric_col(metric)
        model_df = work[[col, "direction", "context", "pair_id"]].dropna().rename(columns={col: "y"})
        if model_df.empty or model_df["direction"].nunique() < 2:
            continue

        fit = cluster_robust_ols(
            model_df,
            formula="y ~ C(direction) * C(context)",
            cluster_col="pair_id",
        )
        if fit is None:
            continue

        for param, coef, pval, se, tval in zip(fit.params.index, fit.params.values, fit.pvalues, fit.bse, fit.tvalues):
            rows.append(
                {
                    "test_type": "cluster_ols",
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "term": param,
                    "coef": float(coef),
                    "se": float(se),
                    "t_or_z": float(tval),
                    "pvalue": float(pval),
                    "n": int(model_df.shape[0]),
                    "r2": float(fit.rsquared),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pvalue_fdr_bh"] = p_adjust_bh(out["pvalue"].to_numpy())
        out["pvalue_holm"] = p_adjust_holm(out["pvalue"].to_numpy())
    return out


def mixed_model_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return pd.DataFrame()

    rows = []
    base = df[df["context"] == "document"].copy()
    if base.empty:
        return pd.DataFrame()

    base["topic_slug"] = base["topic_slug"].fillna("UNKNOWN")
    base["root_lemma"] = base["root_lemma"].fillna("UNK")
    lemma_counts = base["root_lemma"].value_counts()
    keep_lemmas = set(lemma_counts[lemma_counts >= 20].index)
    base["lemma_fe"] = base["root_lemma"].where(base["root_lemma"].isin(keep_lemmas), "OTHER")

    for metric in TARGET_METRICS:
        col = metric_col(metric)
        mdf = base[[col, "direction", "genre", "topic_slug", "doc_name", "lemma_fe"]].dropna().rename(columns={col: "y"})
        if mdf.empty or mdf["direction"].nunique() < 2:
            continue

        fit = None
        note = None
        try:
            fit = smf.mixedlm(
                "y ~ C(direction) + C(genre) + C(topic_slug) + C(lemma_fe)",
                mdf,
                groups=mdf["doc_name"],
                re_formula="1",
            ).fit(method="lbfgs", maxiter=120, disp=False)
            note = "doc_re + genre/topic/lemma FE"
        except Exception:
            try:
                fit = smf.mixedlm(
                    "y ~ C(direction)",
                    mdf,
                    groups=mdf["doc_name"],
                    re_formula="1",
                ).fit(method="lbfgs", maxiter=120, disp=False)
                note = "doc_re + direction"
            except Exception:
                fit = None

        if fit is None:
            continue

        term = next((t for t in fit.params.index if "C(direction)" in t), None)
        if term is None:
            continue
        rows.append(
            {
                "test_type": "mixedlm_document",
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "term": term,
                "coef": float(fit.params[term]),
                "se": float(fit.bse[term]),
                "t_or_z": float(fit.tvalues[term]),
                "pvalue": float(fit.pvalues[term]),
                "n": int(mdf.shape[0]),
                "model": note,
                "converged": bool(getattr(fit, "converged", True)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pvalue_fdr_bh"] = p_adjust_bh(out["pvalue"].to_numpy())
        out["pvalue_holm"] = p_adjust_holm(out["pvalue"].to_numpy())
    return out


def write_paper_tables(confirm: pd.DataFrame, direction: pd.DataFrame, direction_tests: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Paper Tables", ""]

    doc_confirm = confirm[(confirm["subset"] == "changed") & (confirm["context"] == "document")].copy()
    cols = [
        "metric_label",
        "n",
        "mean_norm_delta",
        "mean_ci_low",
        "mean_ci_high",
        "perm_p_two",
        "perm_p_two_fdr_bh",
        "wilcoxon_p_two",
        "wilcoxon_p_two_fdr_bh",
    ]
    lines.append("## Confirmatory: Document Context (Changed)")
    if doc_confirm.empty:
        lines.append("No rows.")
    else:
        lines.append(doc_confirm[cols].round(4).to_markdown(index=False))
    lines.append("")

    doc_dir = direction[(direction["subset"] == "changed") & (direction["context"] == "document")].copy()
    dir_cols = [
        "direction",
        "metric_label",
        "n",
        "mean_norm_delta",
        "mean_ci_low",
        "mean_ci_high",
        "perm_p_two",
        "perm_p_two_fdr_bh_by_dir",
    ]
    lines.append("## Direction Split: Document Context")
    if doc_dir.empty:
        lines.append("No rows.")
    else:
        lines.append(doc_dir[dir_cols].round(4).to_markdown(index=False))
    lines.append("")

    ols = direction_tests[direction_tests["test_type"] == "cluster_ols"].copy()
    lines.append("## Direction x Context (Cluster-Robust OLS)")
    if ols.empty:
        lines.append("No rows.")
    else:
        lines.append(
            ols[["metric_label", "term", "coef", "se", "t_or_z", "pvalue", "pvalue_fdr_bh", "n"]]
            .round(4)
            .to_markdown(index=False)
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_figures(df: pd.DataFrame, confirm: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)

    doc_df = df[df["context"] == "document"].copy()
    if not doc_df.empty:
        plot_rows = []
        for metric in TARGET_METRICS:
            col = metric_col(metric)
            tmp = doc_df[["direction", col]].dropna().copy()
            tmp["metric"] = METRIC_LABELS[metric]
            tmp = tmp.rename(columns={col: "value"})
            plot_rows.append(tmp)
        plot_df = pd.concat(plot_rows, ignore_index=True)

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, x="value", y="metric", hue="direction")
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Normalized Delta (positive = factual better)")
        plt.ylabel("Metric")
        plt.tight_layout()
        plt.savefig(fig_dir / "doc_direction_boxplot.png", dpi=200)
        plt.close()

    heat = confirm[(confirm["subset"] == "changed")].copy()
    if not heat.empty:
        pivot = heat.pivot(index="metric_label", columns="context", values="mean_norm_delta")
        pivot = pivot[[c for c in ALL_CONTEXTS if c in pivot.columns]]

        plt.figure(figsize=(11, 5))
        sns.heatmap(pivot, cmap="coolwarm", center=0.0, annot=True, fmt=".2f")
        plt.title("Mean Normalized Delta by Context")
        plt.tight_layout()
        plt.savefig(fig_dir / "context_metric_heatmap.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confirmatory UID tests.")
    parser.add_argument("--pairs", type=Path, default=Path("analysis/results/pairwise_changed.csv"))
    parser.add_argument("--out_summary", type=Path, default=Path("analysis/results/confirmatory_summary.csv"))
    parser.add_argument("--out_direction", type=Path, default=Path("analysis/results/direction_context_tests.csv"))
    parser.add_argument("--paper_tables", type=Path, default=Path("analysis/results/paper_tables.md"))
    parser.add_argument("--figure_dir", type=Path, default=Path("analysis/results/paper_figures"))
    args = parser.parse_args()

    ensure_dir(args.out_summary.parent)
    ensure_dir(args.figure_dir)

    pair_df = pd.read_csv(args.pairs)
    changed = pair_df.copy()
    high_conf = pair_df[pair_df["high_conf"]].copy()

    confirm_changed = summarize(changed, subset="changed")
    confirm_high = summarize(high_conf, subset="high_conf")
    confirm = pd.concat([confirm_changed, confirm_high], ignore_index=True)

    by_dir_changed = summarize_by_direction(changed, subset="changed")
    by_dir_high = summarize_by_direction(high_conf, subset="high_conf")
    by_direction = pd.concat([by_dir_changed, by_dir_high], ignore_index=True)

    balanced = balanced_direction_downsample(changed)
    ols = direction_context_models(changed)
    mixed = mixed_model_sensitivity(changed)

    direction_tests = pd.concat([by_direction, balanced, ols, mixed], ignore_index=True, sort=False)

    confirm.to_csv(args.out_summary, index=False)
    direction_tests.to_csv(args.out_direction, index=False)

    write_paper_tables(confirm, by_direction, direction_tests, args.paper_tables)
    plot_figures(changed, confirm_changed, args.figure_dir)

    print(f"Wrote: {args.out_summary}")
    print(f"Wrote: {args.out_direction}")
    print(f"Wrote: {args.paper_tables}")
    print(f"Figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
