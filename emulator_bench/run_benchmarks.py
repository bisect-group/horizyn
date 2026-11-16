"""Run all default retrieval splits x seeds: build inputs once per split, then train and evaluate every (split, seed).

Example: python -m emulator_bench.run_benchmarks --gpus 0 1 2 --runs_per_gpu 1
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from .dataset_adapter import discover_splits
from .utils import (
    BASELINE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    HPARAMS,
    run_dir_of,
    split_dir_of,
)


def module_command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", f"emulator_bench.{module}", *map(str, args)]


def run(command: list[str], gpu: str | None) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu or ""}
    return subprocess.run(command, cwd=BASELINE_ROOT, env=env).returncode


def embed_in_shards(label: str, args: argparse.Namespace, out_root: Path) -> None:
    """Embed the proteins of one split with one process per GPU. Each protein is embedded alone, so the vectors do not change."""
    shards = [
        subprocess.Popen(
            module_command(
                "cache_features",
                "--dataset_root",
                args.dataset_root,
                "--out_root",
                out_root,
                "--splits",
                label,
                "--embed_shard",
                f"{i}/{len(args.gpus)}",
                *(["--limit_per_split", args.limit_per_split] if args.limit_per_split else []),
            ),
            cwd=BASELINE_ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
        )
        for i, gpu in enumerate(args.gpus)
    ]
    if any(p.wait() for p in shards):
        raise SystemExit(f"Protein embedding failed for {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Split labels (group/name) or groups. Default: all default splits on disk.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=HPARAMS["seeds"])
    parser.add_argument(
        "--gpus", nargs="+", default=[], help="GPU ids. Each run gets one GPU. No ids: run on CPU."
    )
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on Ampere or newer, fp16 on older GPUs, fp32 on CPU.",
    )
    parser.add_argument("--eval_splits", nargs="+", default=["test"], choices=["val", "test"])
    parser.add_argument("--features_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_per_split", type=int, default=None, help="Smoke test only.")
    parser.add_argument(
        "--embedding_source",
        choices=["prott5", "deterministic"],
        default="prott5",
        help="'deterministic' is for smoke tests only.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Smoke test only.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    splits, skipped = discover_splits(args.dataset_root, args.splits)
    for line in skipped:
        print(f"skipped {line}")
    jobs = [
        (s.label, seed)
        for s in splits
        for seed in args.seeds
        if not all(
            (run_dir_of(out_root, s.group, s.name, seed) / f"results_{e}.json").exists()
            for e in args.eval_splits
        )
    ]
    print(f"{len(splits)} splits, {len(jobs)} runs to do")
    for label, seed in jobs:
        print(f"  {label} seed={seed}")
    if args.dry_run:
        return

    first_gpu = args.gpus[0] if args.gpus else None
    for s in splits:
        if not (split_dir_of(out_root, s.group, s.name) / "data" / "data_summary.json").exists():
            if len(args.gpus) > 1 and args.embedding_source == "prott5":
                embed_in_shards(s.label, args, out_root)
            command = module_command(
                "cache_features",
                "--dataset_root",
                args.dataset_root,
                "--out_root",
                out_root,
                "--splits",
                s.label,
                "--embedding_source",
                args.embedding_source,
                *(["--limit_per_split", args.limit_per_split] if args.limit_per_split else []),
            )
            if run(command, first_gpu):
                raise SystemExit(f"Input build failed for {s.label}")
    if args.features_only:
        return

    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    failed: list[str] = []

    def worker(gpu: str | None) -> None:
        while True:
            try:
                label, seed = work.get_nowait()
            except queue.Empty:
                return
            train = module_command(
                "train",
                "--split",
                label,
                "--seed",
                seed,
                "--out_root",
                out_root,
                *(["--amp"] if args.amp else []),
                *(["--epochs", args.epochs] if args.epochs else []),
            )
            evaluate = module_command(
                "evaluate",
                "--split",
                label,
                "--seed",
                seed,
                "--out_root",
                out_root,
                "--eval_splits",
                *args.eval_splits,
            )
            if run(train, gpu) or run(evaluate, gpu):
                failed.append(f"{label} seed={seed}")

    slots = [str(g) for g in args.gpus for _ in range(args.runs_per_gpu)] or [None]
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise SystemExit("Failed runs:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
