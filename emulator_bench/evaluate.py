"""Evaluate one trained run on the test split (or val).

Two result sets, both from the same checkpoint:
- native: the original scripts/evaluate.py (Top-k hit rates, R-precision, average precision).
- EC accuracy at levels 4..1 for k in 1..50. Each reaction ranks the EC numbers of all proteins of the split
  (train, val and test) by their best protein score. Test reactions dropped for missing features count as misses. Forward and reverse reaction scores are combined with max. Only four-level EC numbers count.
The EC accuracy at k=1 is the number reported in the paper.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from .dataset_adapter import full_ec_labels
from .fit import install_fingerprint_preload
from .results import compute_care_task2_metrics
from .train import build_config
from .utils import (
    BASELINE_ROOT,
    DEFAULT_OUT_ROOT,
    HPARAMS,
    atomic_write,
    run_dir_of,
    split_dir_of,
    write_json,
)

MAX_RANK = max(HPARAMS["evaluation"]["ec_k_values"])


def native_metrics(checkpoint: Path, config_path: Path, device: str) -> dict:
    spec = importlib.util.spec_from_file_location(
        "horizyn_native_evaluate", BASELINE_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_checkpoint(
        str(checkpoint), str(config_path), device, HPARAMS["training"]["retrieval_batch_size"]
    )


def protein_ecs(manifest: pd.DataFrame) -> dict[str, set[str]]:
    ecs: dict[str, set[str]] = defaultdict(set)
    for pid, value in zip(manifest["protein_id"], manifest["ec_number"]):
        ecs[pid].update(full_ec_labels(value))
    return {pid: labels for pid, labels in ecs.items() if labels}


def load_pool(data_dir: Path) -> dict[str, set[str]]:
    """EC numbers of every protein of the split (train, val and test, before any reaction drop)."""
    path = data_dir / "pool_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Rebuild the features of this split.")
    return protein_ecs(pd.read_csv(path))


def rank_ecs(
    model,
    query_fps: np.ndarray,
    target_embeds: torch.Tensor,
    ec_protein_idx: torch.Tensor,
    ec_idx: torch.Tensor,
    num_ecs: int,
    batch: int = 64,
) -> list[list[int]]:
    """Top MAX_RANK EC indices per reaction. A reaction scores an EC by the best score of its proteins, over both directions."""
    ranked = []
    device = target_embeds.device
    with torch.no_grad():
        for start in tqdm(range(0, len(query_fps), batch), desc="EC ranking"):
            fps = torch.from_numpy(query_fps[start : start + batch]).to(
                device
            )  # (B, 2, 2048): forward, reverse
            query = model.model.query_encoder(fps.flatten(0, 1)).view(len(fps), 2, -1)
            scores = (query @ target_embeds.T).amax(dim=1)  # (B, proteins)
            ec_scores = torch.full((len(fps), num_ecs), float("-inf"), device=device)
            ec_scores.scatter_reduce_(
                1,
                ec_idx.unsqueeze(0).expand(len(fps), -1),
                scores[:, ec_protein_idx],
                reduce="amax",
            )
            ranked += ec_scores.topk(min(MAX_RANK, num_ecs), dim=1).indices.tolist()
    return ranked


def add_misses(predictions: pd.DataFrame, data_dir: Path, split: str) -> pd.DataFrame:
    """Append reactions dropped for missing features, with no ranked ECs, so they score as misses."""
    dropped = pd.read_csv(data_dir / f"{split}_dropped.csv")
    misses = pd.DataFrame(
        {
            "reaction_id": dropped["reaction_id"],
            "Reaction": dropped["reaction_smiles"],
            "EC number": dropped["ec_number"].map(lambda v: ";".join(full_ec_labels(v))),
        }
    )
    return pd.concat([predictions, misses[misses["EC number"] != ""]], ignore_index=True)


def evaluate_split(
    model, data_dir: Path, split: str, run_dir: Path, device: str, checkpoint: Path, tag: str
) -> dict:
    from horizyn.datasets.hdf5 import EmbedDataset

    test = pd.read_csv(data_dir / f"{split}_manifest.csv")
    ecs = load_pool(data_dir)
    proteins = EmbedDataset(str(data_dir / "prots_t5.h5"), in_memory=True)
    candidates = [pid for pid in proteins.keys if pid in ecs]
    ec_labels = sorted({ec for pid in candidates for ec in ecs[pid]})
    ec_of = {ec: i for i, ec in enumerate(ec_labels)}
    pairs = [(i, ec_of[ec]) for i, pid in enumerate(candidates) for ec in sorted(ecs[pid])]
    ec_protein_idx = torch.tensor([p for p, _ in pairs], device=device)
    ec_idx = torch.tensor([e for _, e in pairs], device=device)
    with torch.no_grad():
        target_embeds = torch.cat(
            [
                model.model.target_encoder(
                    torch.stack([proteins[k] for k in candidates[i : i + 4096]]).to(device)
                )
                for i in range(0, len(candidates), 4096)
            ]
        )

    truth = {}
    for rid, group in test.groupby("reaction_id", sort=True):
        labels = sorted({ec for value in group["ec_number"] for ec in full_ec_labels(value)})
        if labels:
            truth[rid] = (group["reaction_smiles"].iloc[0], labels)
    with np.load(data_dir / "rxn_fps.npz") as z:
        row = {rid: i for i, rid in enumerate(z["ids"])}
        query_fps = z["fps"][[row[rid] for rid in truth]]
    ranked = rank_ecs(model, query_fps, target_embeds, ec_protein_idx, ec_idx, len(ec_labels))
    predictions = pd.DataFrame(
        [
            {
                "reaction_id": rid,
                "Reaction": smiles,
                "EC number": ";".join(labels),
                **{str(r): ec_labels[j] for r, j in enumerate(top)},
            }
            for (rid, (smiles, labels)), top in zip(truth.items(), ranked)
        ]
    )
    n_dropped = len(pd.read_csv(data_dir / f"{split}_dropped.csv"))
    predictions = add_misses(predictions, data_dir, split)
    atomic_write(
        run_dir / f"predictions_{split}{tag}.csv", lambda tmp: predictions.to_csv(tmp, index=False)
    )

    return {
        "split": split,
        "checkpoint": checkpoint.name,
        "ec_candidates": {"proteins": len(candidates), "ec_numbers": len(ec_labels)},
        "reactions_total": int(test["reaction_id"].nunique()) + n_dropped,
        "reactions_with_full_ec": len(predictions),
        "reactions_missed_unfeaturizable": len(predictions) - len(truth),
        "ec_accuracy": compute_care_task2_metrics(predictions),
    }


def evaluate_run(
    split_label: str,
    seed: int,
    *,
    out_root: Path,
    splits: list[str],
    checkpoint: str = "last",
    device: str = "auto",
) -> None:
    from horizyn.lightning_module import HorizynLitModule

    group, name = split_label.rsplit("/", 1)
    data_dir = split_dir_of(out_root, group, name) / "data"
    run_dir = run_dir_of(out_root, group, name, seed)
    ckpt = run_dir / f"{checkpoint}model.pth"
    tag = "" if checkpoint == "last" else f"_{checkpoint}"
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    install_fingerprint_preload([str(data_dir / "rxn_fps.npz")])
    model = HorizynLitModule.load_from_checkpoint(str(ckpt), map_location=device).eval().to(device)
    for split in splits:
        config_path = run_dir / f"eval_config_{split}.yaml"
        config = build_config(
            data_dir, run_dir, seed, HPARAMS["training"]["max_epochs"], "32-true", eval_split=split
        )
        atomic_write(
            config_path, lambda tmp: Path(tmp).write_text(yaml.safe_dump(config, sort_keys=False))
        )
        result = evaluate_split(model, data_dir, split, run_dir, device, ckpt, tag)
        result["native"] = native_metrics(ckpt, config_path, device)
        result.update(split_label=split_label, seed=seed)
        write_json(run_dir / f"results_{split}{tag}.json", result)
        print(
            f"[emulator_bench] {split_label} seed {seed} {split}: EC k=1 {result['ec_accuracy']['k=1']}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", required=True, help="Split label group/name.")
    parser.add_argument("--seed", type=int, default=HPARAMS["seeds"][0])
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--eval_splits", nargs="+", default=["test"], choices=["val", "test"])
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate_run(
        a.split,
        a.seed,
        out_root=Path(a.out_root),
        splits=a.eval_splits,
        checkpoint=a.checkpoint,
        device=a.device,
    )
