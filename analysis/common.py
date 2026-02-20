from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from conllu import parse_incr
from scipy.stats import wilcoxon

METRICS = [
    "surp_mean",
    "surp_slor",
    "uid_std",
    "uid_mad",
    "uid_pwd",
    "uid_range",
    "uid_slope",
]

PRIMARY_CONTEXTS = [
    "sentence",
    "prev1",
    "prev3",
    "document",
    "sent[-2,+0]",
    "tok[-64,+0]",
]

EXPLORATORY_CONTEXTS = [
    "sent[-2,+2]",
    "tok[-64,+64]",
]

ALL_CONTEXTS = PRIMARY_CONTEXTS + EXPLORATORY_CONTEXTS

METRIC_LABELS = {
    "surp_mean": "Surp. Mean",
    "surp_slor": "SLOR",
    "uid_std": "Surp. STD",
    "uid_mad": "Surp. MAD",
    "uid_pwd": "Local Var.",
    "uid_range": "Surp. Range",
    "uid_slope": "LinReg Slope",
    "uid_slope_abs": "LinReg |Slope|",
}

LOWER_BETTER = {"surp_mean", "uid_std", "uid_mad", "uid_pwd", "uid_range"}
HIGHER_BETTER = {"surp_slor"}

DIRECTION_MAP = {"a>p": "A to P", "p>a": "P to A"}

RED_FLAG_PATTERNS = [
    re.compile(r"\b(am|is|are|was|were)\s+(am|is|are|was|were)\b", re.IGNORECASE),
    re.compile(r"\bby\s+by\b", re.IGNORECASE),
    re.compile(r"\bto\s+is\b", re.IGNORECASE),
    re.compile(r"\bis\s+was\b", re.IGNORECASE),
    re.compile(r"%[A-Za-z]"),
    re.compile(r"\b\w+n['’]t\w+\b"),
    re.compile(r"\b(is|was|are|were)\s+got\b", re.IGNORECASE),
    re.compile(r"\bis\s+lacked\b", re.IGNORECASE),
]

HUMAN_PRONOUNS = {
    "i",
    "me",
    "you",
    "he",
    "him",
    "she",
    "her",
    "we",
    "us",
    "they",
    "them",
    "who",
    "whom",
}

HUMAN_NOUNS = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "children",
    "student",
    "teacher",
    "author",
    "writer",
    "reader",
    "researcher",
    "scientist",
    "doctor",
    "nurse",
    "mother",
    "father",
    "brother",
    "sister",
    "friend",
    "citizen",
    "worker",
    "employee",
    "judge",
    "lawyer",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.startswith("Unnamed")]].copy()


def split_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["doc_id"].astype(str).str.split("::", expand=True)
    out = df.copy()
    out["prefix"] = parts[0]
    out["doc_name"] = parts[1]
    out["conv_idx"] = pd.to_numeric(parts[2], errors="coerce").fillna(-1).astype(int)
    out["conversion"] = parts[3]
    out["direction"] = out["conversion"].map(DIRECTION_MAP)
    split = out["doc_name"].astype(str).str.split("_", n=2, expand=True)
    out["genre"] = split[1].fillna("UNKNOWN")
    out["topic_slug"] = split[2].fillna("UNKNOWN")
    return out


def _has_red_flags(text: str) -> bool:
    if not isinstance(text, str):
        return True
    return any(pattern.search(text) for pattern in RED_FLAG_PATTERNS)


def add_pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in METRICS:
        out[f"d_{metric}"] = out[f"{metric}_f"] - out[f"{metric}_cf"]
        if metric in LOWER_BETTER:
            out[f"d_norm_{metric}"] = out[f"{metric}_cf"] - out[f"{metric}_f"]
        elif metric in HIGHER_BETTER:
            out[f"d_norm_{metric}"] = out[f"{metric}_f"] - out[f"{metric}_cf"]
        else:
            out[f"d_norm_{metric}"] = out[f"d_{metric}"]

    out["uid_slope_abs_f"] = np.abs(out["uid_slope_f"])
    out["uid_slope_abs_cf"] = np.abs(out["uid_slope_cf"])
    out["d_norm_uid_slope_abs"] = out["uid_slope_abs_cf"] - out["uid_slope_abs_f"]

    out["len_ratio"] = out["uid_len_cf"] / out["uid_len_f"].replace(0, np.nan)
    out["changed"] = out["sentence_cf"] != out["sentence_f"]
    out["cf_red_flag"] = out["sentence_cf"].map(_has_red_flags)
    out["length_flag"] = (out["len_ratio"] < 0.85) | (out["len_ratio"] > 1.6)
    out["high_conf"] = out["changed"] & (~out["cf_red_flag"]) & (~out["length_flag"])
    return out


def build_pair_table(uid_df: pd.DataFrame) -> pd.DataFrame:
    uid = split_doc_id(drop_unnamed(uid_df))

    raw_cols = [c for c in ["raw_surps", "raw_uni_surps"] if c in uid.columns]

    factual = uid[uid["prefix"] == "f"].copy()
    cf = uid[uid["prefix"] == "cf"].copy()

    factual_cols = ["doc_name", "sent_idx", "context", "sentence", "uid_len"] + METRICS + raw_cols
    factual = factual[factual_cols].rename(
        columns={
            "sentence": "sentence_f",
            "uid_len": "uid_len_f",
            **{m: f"{m}_f" for m in METRICS},
            **{c: f"{c}_f" for c in raw_cols},
        }
    )

    cf_cols = [
        "doc_id",
        "doc_name",
        "conv_idx",
        "conversion",
        "direction",
        "genre",
        "topic_slug",
        "sent_idx",
        "context",
        "sentence",
        "uid_len",
    ] + METRICS + raw_cols
    cf = cf[cf_cols].rename(
        columns={
            "sentence": "sentence_cf",
            "uid_len": "uid_len_cf",
            **{m: f"{m}_cf" for m in METRICS},
            **{c: f"{c}_cf" for c in raw_cols},
        }
    )

    pair = cf.merge(
        factual,
        on=["doc_name", "sent_idx", "context"],
        how="left",
        validate="many_to_one",
    )
    pair["pair_id"] = (
        pair["doc_name"].astype(str)
        + "::"
        + pair["conv_idx"].astype(str)
        + "::"
        + pair["conversion"].astype(str)
    )
    pair["target_offset"] = pair["sent_idx"] - pair["conv_idx"]
    pair["is_target_sentence"] = pair["target_offset"] == 0

    pair = add_pair_deltas(pair)

    pair["pair_changed"] = pair.groupby("pair_id")["changed"].transform("max")
    pair["pair_high_conf"] = pair.groupby("pair_id")["high_conf"].transform("max")
    return pair


def parse_list_cell(value) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def bootstrap_ci(values: np.ndarray, fn=np.mean, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    sims = fn(vals[idx], axis=1)
    return float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def signflip_test(values: np.ndarray, n_perm: int = 10000, seed: int = 11) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    obs = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, vals.size))
    sims = (signs * vals).mean(axis=1)
    p_two = float(np.mean(np.abs(sims) >= abs(obs)))
    p_greater = float(np.mean(sims >= obs))
    return obs, p_two, p_greater


def wilcoxon_p(values: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if np.allclose(vals, 0.0):
        return 1.0, 1.0
    two = wilcoxon(vals, alternative="two-sided", zero_method="wilcox")
    one = wilcoxon(vals, alternative=alternative, zero_method="wilcox")
    return float(two.pvalue), float(one.pvalue)


def cohen_dz(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return np.nan
    sd = float(np.std(vals, ddof=1))
    if sd == 0:
        return np.nan
    return float(np.mean(vals) / sd)


def p_adjust_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    p = pvals[mask]
    if p.size == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out_mask = np.empty_like(ranked)
    out_mask[order] = adj
    out[mask] = out_mask
    return out


def p_adjust_holm(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    p = pvals[mask]
    if p.size == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    adj = (n - np.arange(n)) * ranked
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out_mask = np.empty_like(ranked)
    out_mask[order] = adj
    out[mask] = out_mask
    return out


def cluster_robust_ols(df: pd.DataFrame, formula: str, cluster_col: str):
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return None

    fit_df = df.dropna().copy()
    if fit_df.empty:
        return None
    model = smf.ols(formula, data=fit_df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": fit_df[cluster_col]})
    return fit


def _subtree_len(token_id: int, children: dict[int, list[int]]) -> int:
    if token_id is None:
        return 0
    seen = set()
    stack = [token_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(children.get(cur, []))
    return len(seen)


def _animacy_class(tok: dict | None) -> str:
    if tok is None:
        return "none"
    lemma = str(tok.get("lemma", "")).lower()
    upos = tok.get("upos")
    if upos == "PRON":
        if lemma in HUMAN_PRONOUNS:
            return "human_pron"
        return "pron"
    if upos == "PROPN":
        return "proper"
    if upos == "NOUN":
        if lemma in HUMAN_NOUNS:
            return "human_noun"
        return "noun"
    return "other"


def _definiteness(tok: dict | None, children_by_id: dict[int, list[dict]]) -> str:
    if tok is None:
        return "none"
    upos = tok.get("upos")
    if upos in {"PRON", "PROPN"}:
        return "definite"

    dets = [c for c in children_by_id.get(tok["id"], []) if c.get("deprel") == "det"]
    det_lemmas = {str(d.get("lemma", "")).lower() for d in dets}
    if det_lemmas & {"the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their"}:
        return "definite"
    if det_lemmas & {"a", "an", "some", "any"}:
        return "indefinite"
    return "bare"


def build_sentence_feature_table(conllu_path: Path) -> pd.DataFrame:
    rows = []
    with conllu_path.open("r", encoding="utf-8") as f:
        current_doc = None
        sent_idx = -1
        prev_lemmas: list[set[str]] = []

        for sent in parse_incr(f):
            md = sent.metadata or {}
            if "newdoc id" in md:
                current_doc = md["newdoc id"]
                sent_idx = 0
                prev_lemmas = []
            else:
                sent_idx += 1

            toks = [tok for tok in sent if isinstance(tok.get("id"), int)]
            id_to_tok = {tok["id"]: tok for tok in toks}
            children: dict[int, list[int]] = {}
            children_full: dict[int, list[dict]] = {}
            for tok in toks:
                head = tok.get("head")
                if not isinstance(head, int):
                    continue
                children.setdefault(head, []).append(tok["id"])
                children_full.setdefault(head, []).append(tok)

            root = next((tok for tok in toks if tok.get("head") == 0), None)
            subj = next((tok for tok in toks if tok.get("deprel") in {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}), None)
            obj = next((tok for tok in toks if tok.get("deprel") in {"obj", "iobj"}), None)

            subj_len = _subtree_len(subj["id"], children) if subj is not None else 0
            obj_len = _subtree_len(obj["id"], children) if obj is not None else 0

            subj_lemma = str(subj.get("lemma", "")).lower() if subj is not None else ""
            obj_lemma = str(obj.get("lemma", "")).lower() if obj is not None else ""

            prev1_lemmas = prev_lemmas[-1] if prev_lemmas else set()
            prev3_lemmas = set().union(*prev_lemmas[-3:]) if prev_lemmas else set()

            subj_dep = abs(subj["id"] - subj["head"]) if subj is not None and isinstance(subj.get("head"), int) else np.nan
            obj_dep = abs(obj["id"] - obj["head"]) if obj is not None and isinstance(obj.get("head"), int) else np.nan
            dep_delta = obj_dep - subj_dep if not (np.isnan(subj_dep) or np.isnan(obj_dep)) else np.nan

            dep_vals = [abs(tok["id"] - tok["head"]) for tok in toks if isinstance(tok.get("head"), int) and tok["head"] > 0]
            mean_dep = float(np.mean(dep_vals)) if dep_vals else np.nan

            has_nsubj_pass = any(tok.get("deprel") == "nsubj:pass" for tok in toks)
            has_agent = any(tok.get("deprel") == "obl:agent" for tok in toks)

            rows.append(
                {
                    "doc_name": current_doc,
                    "sent_idx": sent_idx,
                    "root_lemma": root.get("lemma") if root is not None else None,
                    "root_upos": root.get("upos") if root is not None else None,
                    "sent_len": len(toks),
                    "subj_span_len": subj_len,
                    "obj_span_len": obj_len,
                    "np_len_diff_subj_minus_obj": subj_len - obj_len,
                    "np_len_diff_obj_minus_subj": obj_len - subj_len,
                    "subj_animacy": _animacy_class(subj),
                    "obj_animacy": _animacy_class(obj),
                    "subj_definiteness": _definiteness(subj, children_full),
                    "obj_definiteness": _definiteness(obj, children_full),
                    "subj_given_prev1": float(subj_lemma in prev1_lemmas) if subj_lemma else 0.0,
                    "obj_given_prev1": float(obj_lemma in prev1_lemmas) if obj_lemma else 0.0,
                    "subj_given_prev3": float(subj_lemma in prev3_lemmas) if subj_lemma else 0.0,
                    "obj_given_prev3": float(obj_lemma in prev3_lemmas) if obj_lemma else 0.0,
                    "dep_len_delta_obj_minus_subj": dep_delta,
                    "mean_dep_len": mean_dep,
                    "voice_guess": "passive" if (has_nsubj_pass and has_agent) else "active_or_other",
                }
            )

            cur_lemmas = {str(tok.get("lemma", "")).lower() for tok in toks if tok.get("lemma")}
            prev_lemmas.append(cur_lemmas)

    return pd.DataFrame(rows)


def load_uid_csv(path: Path) -> pd.DataFrame:
    return drop_unnamed(pd.read_csv(path))
