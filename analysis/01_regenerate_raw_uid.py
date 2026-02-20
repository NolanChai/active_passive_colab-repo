from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.uid import run_uid_pipeline

from analysis.common import ALL_CONTEXTS, ensure_dir


def parse_contexts(raw: str | None) -> list[str]:
    if not raw:
        return list(ALL_CONTEXTS)
    txt = str(raw).strip()
    if txt.lower() in {"all", "*"}:
        return list(ALL_CONTEXTS)
    if ";" in txt:
        return [c.strip() for c in txt.split(";") if c.strip()]

    parts = []
    buf = []
    depth = 0
    for ch in txt:
        if ch == "[":
            depth += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth = max(depth - 1, 0)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                parts.append(token)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate UID outputs with raw surprisal traces.")
    parser.add_argument("--conllu", type=Path, default=Path("data/en_gum-ud-train.conllu"))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--contexts", type=str, default="all")
    parser.add_argument("--uid_level", type=str, default="sentence")
    parser.add_argument("--uid_unit", type=str, default="word")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_scope", type=str, default="target", choices=["target", "post", "full"])
    parser.add_argument("--limit_docs", type=int, default=None)
    parser.add_argument("--limit_sents_per_doc", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("analysis/results/uid_raw_word_sentence.csv"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    if args.output.exists() and not args.force:
        print(f"Output exists: {args.output}")
        print("Use --force to overwrite.")
        return

    contexts = parse_contexts(args.contexts)
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
        output_dir=args.output.parent,
        output_file=Path(args.output.name),
        verbose=args.verbose,
        include_raw_surps=True,
        eval_scope=args.eval_scope,
    )

    uid_df.to_csv(args.output, index=False)
    print(f"Wrote {len(uid_df)} rows -> {args.output}")
    print("Columns:", ", ".join(uid_df.columns))


if __name__ == "__main__":
    main()
