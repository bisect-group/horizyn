from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parents[1]
HPARAMS = json.loads((Path(__file__).with_name("default_hparams.json")).read_text())
TASK = HPARAMS["task"]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "datasets" / HPARAMS["dataset_folder"]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "baselines" / "Horizon"
FP_CACHE = "rxnfp_v2"
PROTEIN_CACHE = "prott5_v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reaction_id_for_smiles(smiles: str) -> str:
    return f"rxn_{sha256_text(smiles)[:20]}"


def protein_id_for_sequence(sequence: str) -> str:
    return f"seq_{sha256_text(sequence)[:20]}"


def atomic_write(path: str | Path, write) -> None:
    """Call `write(tmp_path)`, then move the temporary file to `path`. The temp file lives in the target folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: str | Path, data: object) -> None:
    atomic_write(
        path, lambda tmp: Path(tmp).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    )


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def split_dir_of(out_root: str | Path, group: str, name: str) -> Path:
    return Path(out_root) / TASK / group / name


def run_dir_of(out_root: str | Path, group: str, name: str, seed: int) -> Path:
    return split_dir_of(out_root, group, name) / f"seed_{seed}"


def resolve_precision(amp: bool) -> str:
    """Original precision is fp32. With --amp: bf16 on Ampere or newer, fp16 on older GPUs, fp32 on CPU."""
    import torch

    if not amp or not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
