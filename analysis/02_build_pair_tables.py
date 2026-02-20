from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from analysis.common import (
    build_pair_table,
    build_sentence_feature_table,
    ensure_dir,
    load_uid_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired factual/counterfactual UID tables.")
    parser.add_argument("--uid", type=Path, default=Path("analysis/results/uid_raw_word_sentence.csv"))
    parser.add_argument("--conllu", type=Path, default=Path("data/en_gum-ud-train.conllu"))
    parser.add_argument("--output", type=Path, default=Path("analysis/results/pairwise_changed.csv"))
    parser.add_argument("--output_full", type=Path, default=Path("analysis/results/pairwise_full.csv"))
    args = parser.parse_args()

    ensure_dir(args.output.parent)

    uid_df = load_uid_csv(args.uid)
    pair_df = build_pair_table(uid_df)

    sent_features = build_sentence_feature_table(args.conllu)
    pair_df = pair_df.merge(sent_features, on=["doc_name", "sent_idx"], how="left")

    missing_factual = int(pair_df["sentence_f"].isna().sum())
    print(f"Pair integrity: missing factual matches = {missing_factual}")
    if missing_factual > 0:
        pair_df = pair_df[~pair_df["sentence_f"].isna()].copy()

    changed_target = pair_df[(pair_df["is_target_sentence"]) & (pair_df["changed"])].copy()

    pair_df.to_csv(args.output_full, index=False)
    changed_target.to_csv(args.output, index=False)

    summary = pd.DataFrame(
        {
            "rows_full": [len(pair_df)],
            "rows_changed_target": [len(changed_target)],
            "pairs_changed_target": [changed_target["pair_id"].nunique()],
            "a_to_p_rate": [float((changed_target["direction"] == "A to P").mean())],
            "p_to_a_rate": [float((changed_target["direction"] == "P to A").mean())],
            "high_conf_rate": [float(changed_target["high_conf"].mean())],
        }
    )
    print(summary.to_string(index=False))
    print(f"Wrote: {args.output_full}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
