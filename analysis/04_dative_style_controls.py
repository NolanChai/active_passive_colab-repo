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
    build_sentence_feature_table,
    ensure_dir,
    p_adjust_bh,
    p_adjust_holm,
)

TARGET_METRICS = METRICS + ["uid_slope_abs"]


def metric_col(metric: str) -> str:
    return "d_norm_uid_slope_abs" if metric == "uid_slope_abs" else f"d_norm_{metric}"


def fit_cluster_ols(df: pd.DataFrame, formula: str):
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None

    work = df.copy().dropna()
    if work.empty:
        return None

    fit = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["pair_id"]})
    return fit


def prepare_features(df: pd.DataFrame, conllu_path: Path) -> pd.DataFrame:
    needed = {
        "np_len_diff_obj_minus_subj",
        "subj_animacy",
        "obj_animacy",
        "subj_definiteness",
        "obj_definiteness",
        "subj_given_prev1",
        "obj_given_prev1",
        "subj_given_prev3",
        "obj_given_prev3",
        "dep_len_delta_obj_minus_subj",
        "mean_dep_len",
        "root_lemma",
    }
    if needed.issubset(set(df.columns)):
        return df

    feat_df = build_sentence_feature_table(conllu_path)
    return df.merge(feat_df, on=["doc_name", "sent_idx"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dative-style controls for UID deltas.")
    parser.add_argument("--pairs", type=Path, default=Path("analysis/results/pairwise_changed.csv"))
    parser.add_argument("--conllu", type=Path, default=Path("data/en_gum-ud-train.conllu"))
    parser.add_argument("--output", type=Path, default=Path("analysis/results/dative_style_model_results.csv"))
    args = parser.parse_args()

    ensure_dir(args.output.parent)

    df = pd.read_csv(args.pairs)
    df = prepare_features(df, args.conllu)

    df["topic_slug"] = df["topic_slug"].fillna("UNKNOWN")
    df["root_lemma"] = df["root_lemma"].fillna("UNK")
    lemma_counts = df["root_lemma"].value_counts()
    keep_lemmas = set(lemma_counts[lemma_counts >= 20].index)
    df["lemma_fe"] = df["root_lemma"].where(df["root_lemma"].isin(keep_lemmas), "OTHER")

    rows = []

    scopes = {
        "all_contexts": df,
        "document_only": df[df["context"] == "document"].copy(),
    }

    for scope_name, scope_df in scopes.items():
        if scope_df.empty:
            continue

        for metric in TARGET_METRICS:
            y_col = metric_col(metric)
            model_df = scope_df[
                [
                    y_col,
                    "pair_id",
                    "direction",
                    "context",
                    "genre",
                    "topic_slug",
                    "lemma_fe",
                    "np_len_diff_obj_minus_subj",
                    "subj_animacy",
                    "obj_animacy",
                    "subj_definiteness",
                    "obj_definiteness",
                    "subj_given_prev1",
                    "obj_given_prev1",
                    "subj_given_prev3",
                    "obj_given_prev3",
                    "dep_len_delta_obj_minus_subj",
                    "mean_dep_len",
                ]
            ].dropna().copy()
            if model_df.empty or model_df["direction"].nunique() < 2:
                continue

            model_df = model_df.rename(columns={y_col: "y"})

            baseline_formula = "y ~ C(direction) + C(context)"
            full_formula = (
                "y ~ C(direction) + C(context) + "
                "np_len_diff_obj_minus_subj + dep_len_delta_obj_minus_subj + mean_dep_len + "
                "subj_given_prev1 + obj_given_prev1 + subj_given_prev3 + obj_given_prev3 + "
                "C(subj_animacy) + C(obj_animacy) + C(subj_definiteness) + C(obj_definiteness) + "
                "C(genre) + C(topic_slug) + C(lemma_fe) + "
                "C(direction):np_len_diff_obj_minus_subj + C(direction):dep_len_delta_obj_minus_subj"
            )

            base_fit = fit_cluster_ols(model_df, baseline_formula)
            full_fit = fit_cluster_ols(model_df, full_formula)
            if base_fit is None or full_fit is None:
                continue

            dir_term_base = next((t for t in base_fit.params.index if "C(direction)" in t), None)
            dir_term_full = next((t for t in full_fit.params.index if "C(direction)" in t), None)

            rows.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "model": "baseline",
                    "n": int(model_df.shape[0]),
                    "formula": baseline_formula,
                    "r2": float(base_fit.rsquared),
                    "adj_r2": float(base_fit.rsquared_adj),
                    "aic": float(base_fit.aic),
                    "bic": float(base_fit.bic),
                    "direction_term": dir_term_base,
                    "direction_coef": float(base_fit.params[dir_term_base]) if dir_term_base else np.nan,
                    "direction_p": float(base_fit.pvalues[dir_term_base]) if dir_term_base else np.nan,
                }
            )
            rows.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "model": "full_controls",
                    "n": int(model_df.shape[0]),
                    "formula": full_formula,
                    "r2": float(full_fit.rsquared),
                    "adj_r2": float(full_fit.rsquared_adj),
                    "aic": float(full_fit.aic),
                    "bic": float(full_fit.bic),
                    "direction_term": dir_term_full,
                    "direction_coef": float(full_fit.params[dir_term_full]) if dir_term_full else np.nan,
                    "direction_p": float(full_fit.pvalues[dir_term_full]) if dir_term_full else np.nan,
                }
            )
            rows.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "model": "incremental",
                    "n": int(model_df.shape[0]),
                    "formula": "full_controls - baseline",
                    "r2": float(full_fit.rsquared - base_fit.rsquared),
                    "adj_r2": float(full_fit.rsquared_adj - base_fit.rsquared_adj),
                    "aic": float(full_fit.aic - base_fit.aic),
                    "bic": float(full_fit.bic - base_fit.bic),
                    "direction_term": "direction_shift_after_controls",
                    "direction_coef": (
                        float(full_fit.params[dir_term_full] - base_fit.params[dir_term_base])
                        if dir_term_base and dir_term_full
                        else np.nan
                    ),
                    "direction_p": np.nan,
                }
            )

    result = pd.DataFrame(rows)
    p_mask = result["model"].isin(["baseline", "full_controls"])
    result.loc[p_mask, "direction_p_fdr_bh"] = p_adjust_bh(result.loc[p_mask, "direction_p"].to_numpy())
    result.loc[p_mask, "direction_p_holm"] = p_adjust_holm(result.loc[p_mask, "direction_p"].to_numpy())

    result.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
