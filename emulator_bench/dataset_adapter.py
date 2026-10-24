from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .utils import HPARAMS, protein_id_for_sequence, reaction_id_for_smiles

REQUIRED_COLUMNS = {"rxn_smiles", "ec_number", "sequence"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitDir:
    group: str  # path of the split group below the dataset root
    name: str  # "threshold_<t>", or "all" for a flat group
    path: Path

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"


def discover_splits(
    dataset_root: str | Path, requested: list[str] | None = None
) -> tuple[list[SplitDir], list[str]]:
    """Find train/val/test folders on disk. Returns the usable split dirs and the skipped folders with the reason."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    defaults = set(HPARAMS["split_folders"])
    found, skipped = [], []
    for train_file in sorted(root.rglob("train.parquet")):
        path = train_file.parent
        if not all((path / f"{split}.parquet").exists() for split in SPLIT_NAMES):
            continue
        rel = path.relative_to(root)
        if any(part.startswith("backup") for part in rel.parts):
            continue
        threshold = rel.name.startswith("threshold_")
        group = rel.parent.as_posix() if threshold else rel.as_posix()
        split = SplitDir(group, rel.name if threshold else "all", path)
        wanted = (
            any(r in (group, split.label) or group.endswith("/" + r) for r in requested)
            if requested
            else bool(defaults & set(rel.parts))
        )
        if not wanted:
            continue
        if not REQUIRED_COLUMNS <= set(pq.read_schema(train_file).names):
            skipped.append(f"{split.label}: no sequence column")
            continue
        found.append(split)
    return found, skipped


def _labels(value: object) -> list[str]:
    return (
        []
        if pd.isna(value)
        else [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]
    )


def full_ec_labels(value: object) -> list[str]:
    """EC numbers with all four levels known."""
    return sorted(
        {
            ec
            for ec in _labels(value)
            if len(ec.split(".")) == 4 and not {"", "-"} & set(ec.split("."))
        }
    )


def load_records(parquet_path: str | Path, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Read one split file as unique reaction-protein pairs. EC labels of repeated pairs are merged."""
    raw = pd.read_parquet(parquet_path, columns=["rxn_smiles", "sequence", "ec_number"])
    raw_rows = len(raw)
    if limit is not None:
        raw = raw.sample(n=min(limit, raw_rows), random_state=0)
    df = pd.DataFrame(
        {
            "reaction_smiles": raw["rxn_smiles"].astype(str).str.strip(),
            "sequence": raw["sequence"]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.upper(),
            "ec_number": raw["ec_number"],
        }
    )
    parts = df["reaction_smiles"].str.split(">>")
    valid = (
        raw["rxn_smiles"].notna()
        & raw["sequence"].notna()
        & (parts.str.len() == 2)
        & parts.str[0].ne("")
        & parts.str[1].ne("")
        & df["sequence"].ne("")
    )
    df = df[valid]
    rows_valid = len(df)
    df["reaction_id"] = df["reaction_smiles"].map(
        {s: reaction_id_for_smiles(s) for s in df["reaction_smiles"].unique()}
    )
    df["protein_id"] = df["sequence"].map(
        {s: protein_id_for_sequence(s) for s in df["sequence"].unique()}
    )
    df["ec_number"] = df["ec_number"].map(lambda v: ";".join(_labels(v)))
    pair = ["reaction_id", "protein_id"]
    repeated = df.duplicated(pair, keep=False)
    merged = (
        df[repeated]
        .groupby(pair, as_index=False)
        .agg(
            reaction_smiles=("reaction_smiles", "first"),
            sequence=("sequence", "first"),
            ec_number=(
                "ec_number",
                lambda v: ";".join(sorted({e for x in v for e in _labels(x)})),
            ),
        )
    )
    df = pd.concat([df[~repeated], merged]).sort_values(pair).reset_index(drop=True)
    stats = {
        "raw_rows": int(raw_rows),
        "sampled_rows": int(len(raw)),
        "invalid_rows": int(len(raw) - rows_valid),
        "unique_pairs": int(len(df)),
    }
    return df, stats
