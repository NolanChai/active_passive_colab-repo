from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

from analysis.common import ensure_dir, parse_list_cell


def detrend(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    if vals.size <= 1:
        return vals
    x = np.arange(vals.size, dtype=float)
    slope, intercept = np.polyfit(x, vals, 1)
    return vals - (slope * x + intercept)


def z_norm(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        return vals
    sd = vals.std()
    if sd == 0:
        return np.zeros_like(vals)
    return (vals - vals.mean()) / sd


def to_interp(arr: np.ndarray, n_points: int = 128) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        return vals
    if vals.size == 1:
        return np.repeat(vals[0], n_points)
    xp = np.arange(vals.size, dtype=float)
    xq = np.linspace(0, vals.size - 1, n_points)
    return np.interp(xq, xp, vals)


def spike_features(trace: np.ndarray, factor: float) -> dict:
    vals = np.asarray(trace, dtype=float)
    if vals.size == 0:
        return {
            "peak_count": np.nan,
            "max_peak_height": np.nan,
            "supra_area": np.nan,
            "mean_inter_peak_interval": np.nan,
            "burstiness": np.nan,
            "threshold": np.nan,
        }

    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-8
    thr = med + factor * mad

    if vals.size >= 3:
        peak_mask = (vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:]) & (vals[1:-1] > thr)
        peak_idx = np.where(peak_mask)[0] + 1
    else:
        peak_idx = np.array([], dtype=int)

    intervals = np.diff(peak_idx)
    if intervals.size > 0:
        mu = intervals.mean()
        sd = intervals.std()
        burst = (sd - mu) / (sd + mu + 1e-8)
    else:
        burst = np.nan

    if peak_idx.size > 0:
        max_peak = float(np.max(vals[peak_idx] - thr))
        mean_ipi = float(intervals.mean()) if intervals.size > 0 else np.nan
    else:
        max_peak = 0.0
        mean_ipi = np.nan

    return {
        "peak_count": float(peak_idx.size),
        "max_peak_height": max_peak,
        "supra_area": float(np.clip(vals - thr, 0, None).sum()),
        "mean_inter_peak_interval": mean_ipi,
        "burstiness": float(burst) if not np.isnan(burst) else np.nan,
        "threshold": float(thr),
    }


def spectral_features(trace: np.ndarray) -> dict:
    vals = np.asarray(trace, dtype=float)
    if vals.size < 2:
        return {
            "spectral_centroid": np.nan,
            "dominant_frequency": np.nan,
            "spectral_entropy": np.nan,
            "low_high_band_ratio": np.nan,
            "harmonic_ratio": np.nan,
        }

    fft = np.fft.rfft(vals)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(vals.size, d=1.0)
    eps = 1e-12

    total = power.sum() + eps
    centroid = float((freqs * power).sum() / total)

    if power.size > 1:
        fund_idx = int(np.argmax(power[1:]) + 1)
    else:
        fund_idx = 0
    dom = float(freqs[fund_idx]) if fund_idx < freqs.size else np.nan

    p = power / total
    entropy = -np.sum(p * np.log2(p + eps))
    entropy = float(entropy / np.log2(len(p))) if len(p) > 1 else 0.0

    cut = max(1, int(0.25 * len(power)))
    low = power[1:cut].sum() if cut > 1 else 0.0
    high = power[cut:].sum()
    low_high = float(low / (high + eps))

    harm_idx = min(2 * fund_idx, power.size - 1) if fund_idx > 0 else 0
    harm_ratio = float(power[harm_idx] / (power[fund_idx] + eps)) if fund_idx > 0 else np.nan

    return {
        "spectral_centroid": centroid,
        "dominant_frequency": dom,
        "spectral_entropy": entropy,
        "low_high_band_ratio": low_high,
        "harmonic_ratio": harm_ratio,
    }


def feature_block(trace: np.ndarray, prefix: str) -> dict:
    out = {}
    dt = detrend(trace)
    zn = z_norm(dt)

    for factor, label in [(2.0, "t20"), (2.5, "t25"), (3.0, "t30")]:
        feats = spike_features(zn, factor)
        for key, value in feats.items():
            out[f"{prefix}_{key}_{label}"] = value

    spec = spectral_features(zn)
    for key, value in spec.items():
        out[f"{prefix}_{key}"] = value

    inter = to_interp(zn, n_points=128)
    for factor, label in [(2.0, "t20"), (2.5, "t25"), (3.0, "t30")]:
        feats = spike_features(inter, factor)
        for key, value in feats.items():
            out[f"{prefix}_interp128_{key}_{label}"] = value

    spec_i = spectral_features(inter)
    for key, value in spec_i.items():
        out[f"{prefix}_interp128_{key}"] = value

    out[f"{prefix}_len"] = int(len(trace))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spike + harmonic analysis on raw surprisal traces.")
    parser.add_argument("--pairs", type=Path, default=Path("analysis/results/pairwise_changed.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/results/signal_metrics.csv"))
    args = parser.parse_args()

    ensure_dir(args.output.parent)

    df = pd.read_csv(args.pairs)
    required = {"raw_surps_f", "raw_surps_cf"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            "Missing raw trace columns in pair table. Re-run 01_regenerate_raw_uid.py and 02_build_pair_tables.py first."
        )

    rows = []
    for row in df.itertuples(index=False):
        trace_f = np.asarray(parse_list_cell(getattr(row, "raw_surps_f")), dtype=float)
        trace_cf = np.asarray(parse_list_cell(getattr(row, "raw_surps_cf")), dtype=float)
        if trace_f.size < 2 or trace_cf.size < 2:
            continue

        base = {
            "pair_id": row.pair_id,
            "doc_name": row.doc_name,
            "sent_idx": int(row.sent_idx),
            "direction": row.direction,
            "context": row.context,
        }

        feats_f = feature_block(trace_f, "f")
        feats_cf = feature_block(trace_cf, "cf")

        out = base | feats_f | feats_cf

        for key in list(feats_f.keys()):
            cf_key = key.replace("f_", "cf_", 1)
            if cf_key in feats_cf:
                out[f"delta_{key[2:]}"] = feats_cf[cf_key] - feats_f[key]

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}")
    print(f"Rows: {len(out_df)}")
    if not out_df.empty:
        sample_cols = [c for c in out_df.columns if c.startswith("delta_peak_count") or c.startswith("delta_spectral_entropy")]
        if sample_cols:
            print(out_df[sample_cols].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
