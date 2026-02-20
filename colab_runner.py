#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ALL_CONTEXTS_ARG = "sentence;prev1;prev3;document;sent[-2,+0];sent[-2,+2];tok[-64,+0];tok[-64,+64]"


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


def ensure_pip() -> None:
    if shutil.which("pip"):
        return
    run([sys.executable, "-m", "ensurepip", "--upgrade"])


def install_requirements(source_dir: Path) -> None:
    ensure_pip()
    req = source_dir / "requirements.txt"
    if not req.exists():
        return
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=source_dir)


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


def ensure_gum_data(source_dir: Path) -> None:
    data_dir = source_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "en_gum-ud-train.conllu"
    if target.exists():
        return

    urls = [
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-train.conllu",
        "https://raw.githubusercontent.com/NolanChai/active-passive-alternations/main/data/en_gum-ud-train.conllu",
    ]
    for url in urls:
        try:
            print(f"Downloading GUM data from: {url}")
            urllib.request.urlretrieve(url, target)
            print(f"Saved: {target}")
            return
        except Exception as e:
            print(f"[warn] Failed download: {e}")
            continue
    raise RuntimeError("Unable to download en_gum-ud-train.conllu from known sources.")


def prepare_source(
    source_mode: str,
    repo_url: str,
    branch: str,
    source_dir: Path,
    skip_install: bool = False,
) -> None:
    if source_mode == "git":
        clone_or_update(repo_url, branch, source_dir)
    if not skip_install:
        install_requirements(source_dir)
    ensure_gum_data(source_dir)


def ensure_cf_word_uid(source_dir: Path, model: str, force: bool = False) -> None:
    out = source_dir / "outputs" / "cf_word_sentence_uid.csv"
    if out.exists() and not force:
        return
    (source_dir / "outputs").mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "run_uid_pipeline.py",
            "data",
            model,
            "--generate_counterfactual",
            "--uid_level",
            "sentence",
            "--uid_unit",
            "word",
            "--output_dir",
            "outputs",
            "--output_name",
            "cf_word_sentence_uid",
            "--include_raw_surps",
            "false",
            "--eval_scope",
            "target",
        ],
        cwd=source_dir,
    )


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


def run_confirmatory(source_dir: Path, model: str, regenerate_uid: bool) -> None:
    ensure_cf_word_uid(source_dir, model=model, force=regenerate_uid)
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
            sys.executable,
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
    run([sys.executable, "analysis/03_confirmatory_tests.py"], cwd=source_dir)
    run([sys.executable, "analysis/04_dative_style_controls.py"], cwd=source_dir)
    run([sys.executable, "analysis/06_genre_topic_models.py"], cwd=source_dir)


def run_raw_signal(source_dir: Path, model: str, limit_docs: int | None, full: bool, force: bool = False) -> None:
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
        sys.executable,
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
    ]
    if limit_docs is not None:
        cmd.extend(["--limit_docs", str(limit_docs)])
    if force:
        cmd.append("--force")

    run(cmd, cwd=source_dir)

    run(
        [
            sys.executable,
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
            sys.executable,
            "analysis/05_signal_spike_harmonic.py",
            "--pairs",
            pair_out,
            "--output",
            signal_out,
        ],
        cwd=source_dir,
    )


def run_impulse(
    source_dir: Path,
    model: str,
    limit_docs: int | None,
    k: int,
    full: bool,
    force: bool = False,
) -> None:
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
        sys.executable,
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
    ]
    if limit_docs is not None:
        cmd.extend(["--limit_docs", str(limit_docs)])
    if force:
        cmd.append("--force")

    run(cmd, cwd=source_dir)


def run_uid_cache(
    source_dir: Path,
    model: str,
    eval_scope: str,
    output_name: str,
    force: bool,
) -> None:
    out = source_dir / "outputs" / f"{output_name}.csv"
    if out.exists() and not force:
        return
    cmd = [
        sys.executable,
        "run_uid_pipeline.py",
        "data",
        model,
        "--generate_counterfactual",
        "--uid_level",
        "sentence",
        "--uid_unit",
        "word",
        "--context",
        ALL_CONTEXTS_ARG,
        "--include_raw_surps",
        "true",
        "--eval_scope",
        eval_scope,
        "--output_dir",
        "outputs",
        "--output_name",
        output_name,
    ]
    run(cmd, cwd=source_dir)


def run_cache_bundle(source_dir: Path, model: str, include_full_scope: bool, force: bool) -> None:
    run_uid_cache(
        source_dir=source_dir,
        model=model,
        eval_scope="target",
        output_name="uid_cache_all_ctx_target_raw",
        force=force,
    )
    run_uid_cache(
        source_dir=source_dir,
        model=model,
        eval_scope="post",
        output_name="uid_cache_all_ctx_post_raw",
        force=force,
    )
    if include_full_scope:
        run_uid_cache(
            source_dir=source_dir,
            model=model,
            eval_scope="full",
            output_name="uid_cache_all_ctx_full_raw",
            force=force,
        )


def persist_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-passive UID analyses on Colab.")
    parser.add_argument("--source-mode", default="local", choices=["local", "git"])
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
            "cache_only",
            "all_sample",
            "all_full",
            "all_full_cached",
        ],
    )
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--limit-docs", type=int, default=40, help="Used by sample profiles.")
    parser.add_argument("--k", type=int, default=10, help="Offset horizon for impulse profile.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install -r requirements.txt")
    parser.add_argument(
        "--cache-include-full-scope",
        action="store_true",
        help="Cache eval_scope=full UID tables in addition to target and post.",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Force regeneration for cache output files.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Optional path (e.g., Google Drive) to copy outputs and analysis/results after run.",
    )
    parser.add_argument(
        "--regenerate-confirmatory-uid",
        action="store_true",
        help="Force regeneration of outputs/cf_word_sentence_uid.csv before confirmatory runs.",
    )

    args = parser.parse_args()

    workspace = Path.cwd()
    source_dir = workspace if args.source_mode == "local" else workspace / args.source_dir

    prepare_source(
        args.source_mode,
        args.source_repo,
        args.source_branch,
        source_dir,
        skip_install=args.skip_install,
    )

    if args.profile == "prepare":
        print("Prepared source repository. No analysis profile run.")
        return
    if args.profile == "doctor":
        print("Environment check:")
        print(f"  - source_mode: {args.source_mode}")
        print(f"  - source_dir: {source_dir}")
        print(f"  - python: {sys.executable}")
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
        run_confirmatory(
            source_dir,
            model=args.model,
            regenerate_uid=args.regenerate_confirmatory_uid,
        )
    elif args.profile == "raw_signal_sample":
        run_raw_signal(source_dir, args.model, args.limit_docs, full=False)
    elif args.profile == "impulse_sample":
        run_impulse(source_dir, args.model, args.limit_docs, args.k, full=False)
    elif args.profile == "full_raw_signal":
        run_raw_signal(source_dir, args.model, None, full=True, force=args.force_cache)
    elif args.profile == "full_impulse":
        run_impulse(source_dir, args.model, None, args.k, full=True, force=args.force_cache)
    elif args.profile == "cache_only":
        run_cache_bundle(
            source_dir=source_dir,
            model=args.model,
            include_full_scope=args.cache_include_full_scope,
            force=args.force_cache,
        )
    elif args.profile == "all_sample":
        run_confirmatory(
            source_dir,
            model=args.model,
            regenerate_uid=args.regenerate_confirmatory_uid,
        )
        run_raw_signal(source_dir, args.model, args.limit_docs, full=False)
        run_impulse(source_dir, args.model, args.limit_docs, args.k, full=False)
    elif args.profile == "all_full":
        run_confirmatory(
            source_dir,
            model=args.model,
            regenerate_uid=args.regenerate_confirmatory_uid,
        )
        run_raw_signal(source_dir, args.model, None, full=True, force=args.force_cache)
        run_impulse(source_dir, args.model, None, args.k, full=True, force=args.force_cache)
    elif args.profile == "all_full_cached":
        run_confirmatory(
            source_dir,
            model=args.model,
            regenerate_uid=args.regenerate_confirmatory_uid,
        )
        run_raw_signal(source_dir, args.model, None, full=True, force=args.force_cache)
        run_impulse(source_dir, args.model, None, args.k, full=True, force=args.force_cache)
        run_cache_bundle(
            source_dir=source_dir,
            model=args.model,
            include_full_scope=args.cache_include_full_scope,
            force=args.force_cache,
        )

    if args.persist_dir:
        persist_root = Path(args.persist_dir)
        persist_tree(source_dir / "outputs", persist_root / "outputs")
        persist_tree(source_dir / "analysis" / "results", persist_root / "analysis_results")
        print(f"Copied outputs to cache path: {persist_root}")

    print("Done. See outputs under:")
    print(source_dir / "analysis/results")


if __name__ == "__main__":
    main()
