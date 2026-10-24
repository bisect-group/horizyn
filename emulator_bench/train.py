"""Train Horizyn on one split and seed with the original train.py (through emulator_bench.fit).

Run folder files: config.yaml, logs/, bestmodel.pth (best val/loss), lastmodel.pth (last epoch),
checkpoint_last.pt (resumable Lightning state), train_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from .utils import (
    BASELINE_ROOT,
    DEFAULT_OUT_ROOT,
    HPARAMS,
    atomic_write,
    resolve_precision,
    run_dir_of,
    split_dir_of,
    write_json,
)

TRAIN = HPARAMS["training"]
RXN = HPARAMS["reaction_features"]


def build_config(
    data_dir: Path, run_dir: Path, seed: int, epochs: int, precision: str, eval_split: str = "val"
) -> dict:
    path = lambda name: str((data_dir / name).resolve())  # noqa: E731
    model = HPARAMS["model"]
    return {
        "seed": seed,
        "logging": {
            "log_dir": str(run_dir / "logs"),
            "checkpoint_dir": str(run_dir / "lightning"),
            "save_every_n_epochs": min(TRAIN["save_every_n_epochs"], epochs),
            "log_every_n_steps": 1,
        },
        "data": {
            "train_pairs_path": path("train_pairs.csv"),
            "train_reactions_path": path("train_rxns.csv"),
            "test_pairs_path": path(
                f"{eval_split}_pairs.csv"
            ),  # the original names its validation split "test"
            "test_reactions_path": path(f"{eval_split}_rxns.csv"),
            "protein_embeds_path": path("prots_t5.h5"),
            "train_batch_size": TRAIN["train_batch_size"],
            "retrieval_batch_size": TRAIN["retrieval_batch_size"],
            "num_workers": 0,  # fit.py builds batches in the main process
            "pin_memory": False,
            "rdkit_fp_dim": RXN["rdkit_plus"]["vec_dim"],
            "drfp_dim": RXN["drfp"]["vec_dim"],
            "standardize_reactions": RXN["standardize"]["enabled"],
            **{f"standardize_{k}": v for k, v in RXN["standardize"].items() if k != "enabled"},
        },
        "model": {
            "name": "DualContrastiveModel",
            "query_encoder_dims": model["query_encoder_dims"],
            "target_encoder_dims": model["target_encoder_dims"],
            "embedding_dim": model["embedding_dim"],
        },
        "training": {
            "max_epochs": epochs,
            "learning_rate": TRAIN["learning_rate"],
            "weight_decay": TRAIN["weight_decay"],
            "precision": precision,
            "loss": {**TRAIN["loss"], "beta_min": 0.01, "beta_max": 100.0},
            "metrics": {"top_k": TRAIN["metrics_top_k"]},
            "check_val_every_n_epoch": min(TRAIN["check_val_every_n_epoch"], epochs),
            "enable_progress_bar": False,
        },
    }


def export_checkpoints(run_dir: Path, epochs: int) -> dict:
    """Write bestmodel.pth and lastmodel.pth (weights and hyper-parameters, no optimizer state).

    The original train.py exits normally on KeyboardInterrupt, so the last checkpoint must reach the final epoch.
    lastmodel.pth is written last: it marks the export as complete."""
    lightning = run_dir / "lightning"
    last_path = (
        lightning / "last.ckpt"
        if (lightning / "last.ckpt").exists()
        else run_dir / "checkpoint_last.pt"
    )
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(last["epoch"]) + 1 != epochs:
        raise RuntimeError(f"Training stopped at epoch {last['epoch'] + 1} of {epochs}")
    best_name = next(
        (
            Path(s["best_model_path"]).name
            for s in last.get("callbacks", {}).values()
            if s.get("best_model_path")
        ),
        None,
    )
    best = (
        torch.load(lightning / best_name, map_location="cpu", weights_only=False)
        if best_name and (lightning / best_name).exists()
        else last
    )
    for name, ckpt in (("bestmodel.pth", best), ("lastmodel.pth", last)):
        slim = {
            k: ckpt[k]
            for k in (
                "state_dict",
                "hyper_parameters",
                "pytorch-lightning_version",
                "epoch",
                "global_step",
            )
        }
        atomic_write(run_dir / name, lambda tmp, c=slim: torch.save(c, tmp))
    if last_path.parent == lightning:
        os.replace(last_path, run_dir / "checkpoint_last.pt")
    shutil.rmtree(lightning, ignore_errors=True)
    return {"last_epoch": int(last["epoch"]), "best_epoch": int(best["epoch"])}


def train_run(
    split_label: str, seed: int, *, out_root: Path, amp: bool = False, epochs: int | None = None
) -> None:
    group, name = split_label.rsplit("/", 1)
    data_dir = split_dir_of(out_root, group, name) / "data"
    run_dir = run_dir_of(out_root, group, name, seed)
    if all((run_dir / n).exists() for n in ("lastmodel.pth", "train_summary.json")):
        print(f"[emulator_bench] {split_label} seed {seed}: already trained", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    precision = resolve_precision(amp)
    epochs = epochs or TRAIN["max_epochs"]
    config_path = run_dir / "config.yaml"
    atomic_write(
        config_path,
        lambda tmp: Path(tmp).write_text(
            yaml.safe_dump(
                build_config(data_dir, run_dir, seed, epochs, precision), sort_keys=False
            )
        ),
    )

    resume = next(
        (
            p
            for p in (run_dir / "lightning" / "last.ckpt", run_dir / "checkpoint_last.pt")
            if p.exists()
        ),
        None,
    )
    command = [
        sys.executable,
        "-m",
        "emulator_bench.fit",
        "--fp_file",
        str(data_dir / "rxn_fps.npz"),
        "--",
        "--config",
        str(config_path),
        "--seed",
        str(seed),
    ]
    if resume:
        command += ["--resume", str(resume)]
    start = time.time()
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }  # lowers the reserved GPU memory
    subprocess.run(command, cwd=BASELINE_ROOT, env=env, check=True)
    info = export_checkpoints(run_dir, epochs)
    write_json(
        run_dir / "train_summary.json",
        {
            "split": split_label,
            "seed": seed,
            "precision": precision,
            "epochs": epochs,
            "train_seconds": round(time.time() - start, 1),
            **info,
            "data": json.loads((data_dir / "data_summary.json").read_text())["stats"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        required=True,
        help="Split label group/name, for example uniprot_time_splits/all.",
    )
    parser.add_argument("--seed", type=int, default=HPARAMS["seeds"][0])
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on Ampere or newer, fp16 on older GPUs, fp32 on CPU.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Smoke test only.")
    return parser.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train_run(a.split, a.seed, out_root=Path(a.out_root), amp=a.amp, epochs=a.epochs)
