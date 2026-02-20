#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"$ {' '.join(cmd)}{where}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_capture(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )


def ensure_uv() -> None:
    if shutil.which("uv"):
        return
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])


def branch_exists(repo_url: str, branch: str) -> bool:
    cp = run_capture(["git", "ls-remote", "--heads", repo_url, branch])
    return cp.returncode == 0 and bool(cp.stdout.strip())


def clone_or_update(repo_url: str, branch: str, dest: Path) -> None:
    branch_available = branch_exists(repo_url, branch)

    if not dest.exists():
        if branch_available:
            run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(dest)])
        else:
            print(f"[warn] Branch '{branch}' not found. Cloning default branch instead.")
            run(["git", "clone", "--depth", "1", repo_url, str(dest)])
        return

    run(["git", "fetch", "--all"], cwd=dest)
    if branch_available:
        run(["git", "checkout", branch], cwd=dest)
        run(["git", "pull", "--ff-only", "origin", branch], cwd=dest)
    else:
        print(f"[warn] Branch '{branch}' not found. Keeping existing checked-out branch.")


def prepare_source(repo_url: str, branch: str, source_dir: Path) -> None:
    ensure_uv()
    clone_or_update(repo_url, branch, source_dir)
    run(["uv", "sync"], cwd=source_dir)


def require_paths(source_dir: Path, rel_paths: list[str]) -> None:
    missing = [p for p in rel_paths if not (source_dir / p).exists()]
    if not missing:
        return
    missing_text = "\n".join(f"  - {m}" for m in missing)
    raise FileNotFoundError(
        "Source repo is missing required files for this profile:\n"
        f"{missing_text}\n\n"
        "Point --source-repo/--source-branch to a branch that includes the analysis scripts."
    )


def run_confirmatory(source_dir: Path) -> None:
    require_paths(
        source_dir,
        [
            "analysis/02_build_pair_tables.py",
            "analysis/03_confirmatory_tests.py",
            "analysis/04_dative_style_controls.py",
            "analysis/06_genre_topic_models.py",
            "outputs/cf_word_sentence_uid.csv",
            "data/en_gum-ud-train.conllu",
        ],
    )

    run(
        [
            "uv",
            "run",
            "python",
            "analysis/02_build_pair_tables.py",
            "--uid",
            "outputs/cf_word_sentence_uid.csv",
            "--output",
            "analysis/results/pairwise_changed.csv",
            "--output_full",
            "analysis/results/pairwise_full.csv",
        ],
        cwd=source_dir,
    )
    run(["uv", "run", "python", "analysis/03_confirmatory_tests.py"], cwd=source_dir)
    run(["uv", "run", "python", "analysis/04_dative_style_controls.py"], cwd=source_dir)
    run(["uv", "run", "python", "analysis/06_genre_topic_models.py"], cwd=source_dir)


def run_raw_signal(source_dir: Path, model: str, limit_docs: int | None, full: bool) -> None:
    require_paths(
        source_dir,
        [
            "analysis/01_regenerate_raw_uid.py",
            "analysis/02_build_pair_tables.py",
            "analysis/05_signal_spike_harmonic.py",
            "data/en_gum-ud-train.conllu",
        ],
    )

    uid_out = "analysis/results/uid_raw_word_sentence.csv" if full else "analysis/results/uid_raw_word_sentence_sample.csv"
    pair_out = "analysis/results/pairwise_changed.csv" if full else "analysis/results/pairwise_changed_raw_sample.csv"
    pair_full = "analysis/results/pairwise_full.csv" if full else "analysis/results/pairwise_full_raw_sample.csv"
    signal_out = "analysis/results/signal_metrics.csv" if full else "analysis/results/signal_metrics_sample.csv"

    cmd = [
        "uv",
        "run",
        "python",
        "analysis/01_regenerate_raw_uid.py",
        "--conllu",
        "data/en_gum-ud-train.conllu",
        "--model",
        model,
        "--contexts",
        "document",
        "--output",
        uid_out,
        "--eval_scope",
        "target",
        "--force",
    ]
    if limit_docs is not None:
        cmd.extend(["--limit_docs", str(limit_docs)])

    run(cmd, cwd=source_dir)

    run(
        [
            "uv",
            "run",
            "python",
            "analysis/02_build_pair_tables.py",
            "--uid",
            uid_out,
            "--output",
            pair_out,
            "--output_full",
            pair_full,
        ],
        cwd=source_dir,
    )

    run(
        [
            "uv",
            "run",
            "python",
            "analysis/05_signal_spike_harmonic.py",
            "--pairs",
            pair_out,
            "--output",
            signal_out,
        ],
        cwd=source_dir,
    )


def run_impulse(source_dir: Path, model: str, limit_docs: int | None, k: int, full: bool) -> None:
    require_paths(
        source_dir,
        [
            "analysis/07_propagation_impulse.py",
            "data/en_gum-ud-train.conllu",
        ],
    )

    raw_out = "analysis/results/uid_post_word_sentence.csv" if full else "analysis/results/uid_post_word_sentence_sample.csv"
    summary_out = (
        "analysis/results/impulse_response_summary.csv"
        if full
        else "analysis/results/impulse_response_summary_sample.csv"
    )

    cmd = [
        "uv",
        "run",
        "python",
        "analysis/07_propagation_impulse.py",
        "--conllu",
        "data/en_gum-ud-train.conllu",
        "--model",
        model,
        "--contexts",
        "document",
        "--k",
        str(k),
        "--raw_output",
        raw_out,
        "--summary_output",
        summary_out,
        "--force",
    ]
    if limit_docs is not None:
        cmd.extend(["--limit_docs", str(limit_docs)])

    run(cmd, cwd=source_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-passive UID analyses on Colab.")
    parser.add_argument(
        "--source-repo",
        default="https://github.com/NolanChai/active-passive-alternations.git",
        help="Source repository containing analysis scripts.",
    )
    parser.add_argument("--source-branch", default="main")
    parser.add_argument("--source-dir", default="active-passive-alternations")
    parser.add_argument(
        "--profile",
        default="confirmatory",
        choices=[
            "doctor",
            "prepare",
            "confirmatory",
            "raw_signal_sample",
            "impulse_sample",
            "full_raw_signal",
            "full_impulse",
        ],
    )
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--limit-docs", type=int, default=40, help="Used by sample profiles.")
    parser.add_argument("--k", type=int, default=10, help="Offset horizon for impulse profile.")

    args = parser.parse_args()

    workspace = Path.cwd()
    source_dir = workspace / args.source_dir

    prepare_source(args.source_repo, args.source_branch, source_dir)

    if args.profile == "prepare":
        print("Prepared source repository. No analysis profile run.")
        return
    if args.profile == "doctor":
        print("Environment check:")
        print(f"  - source_dir: {source_dir}")
        print(f"  - uv: {shutil.which('uv') is not None}")
        print("  - key files:")
        for rel in [
            "analysis/01_regenerate_raw_uid.py",
            "analysis/02_build_pair_tables.py",
            "analysis/03_confirmatory_tests.py",
            "analysis/04_dative_style_controls.py",
            "analysis/05_signal_spike_harmonic.py",
            "analysis/06_genre_topic_models.py",
            "analysis/07_propagation_impulse.py",
            "src/uid.py",
            "run_uid_pipeline.py",
            "outputs/cf_word_sentence_uid.csv",
            "data/en_gum-ud-train.conllu",
        ]:
            print(f"    {'OK ' if (source_dir / rel).exists() else 'MISS'} {rel}")
        return

    if args.profile == "confirmatory":
        run_confirmatory(source_dir)
    elif args.profile == "raw_signal_sample":
        run_raw_signal(source_dir, args.model, args.limit_docs, full=False)
    elif args.profile == "impulse_sample":
        run_impulse(source_dir, args.model, args.limit_docs, args.k, full=False)
    elif args.profile == "full_raw_signal":
        run_raw_signal(source_dir, args.model, None, full=True)
    elif args.profile == "full_impulse":
        run_impulse(source_dir, args.model, None, args.k, full=True)

    print("Done. See outputs under:")
    print(source_dir / "analysis/results")


if __name__ == "__main__":
    main()
