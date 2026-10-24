"""Build the inputs of one split: ProtT5 vectors, reaction fingerprints, pair CSVs and the protein HDF5 file.

Reaction fingerprints come from the original Horizyn dataset classes. Protein vectors are cached once per
unique sequence. Both caches live under <out_root>/embeddings/<kind>_v<N>/.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import uuid
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset_adapter import SPLIT_NAMES, SplitDir, discover_splits, full_ec_labels, load_records
from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    FP_CACHE,
    HPARAMS,
    PROTEIN_CACHE,
    atomic_write,
    read_json,
    reaction_id_for_smiles,
    sha256_text,
    split_dir_of,
    write_json,
)

RXN = HPARAMS["reaction_features"]
PROT = HPARAMS["protein_features"]
STD = {f"standardize_{k}": v for k, v in RXN["standardize"].items() if k != "enabled"}
FP_DIM = RXN["rdkit_plus"]["vec_dim"]
SMOKE_CACHE = "smoke_embeddings"
SHARD_ROWS = 20000


# ---------------------------------------------------------------------------------------------------------
# Cache folders
# ---------------------------------------------------------------------------------------------------------
def cache_dir(out_root: Path, name: str, config: dict) -> Path:
    """Return the cache folder. The folder keeps its config. A different config needs a new version number."""
    path = Path(out_root) / "embeddings" / name
    config_path = path / "config.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise ValueError(
                f"{config_path} holds a different config. Bump the cache version in utils.py."
            )
    else:
        write_json(config_path, config)
    return path


def item_path(root: Path, key: str, suffix: str) -> Path:
    return root / key.split("_", 1)[-1][:2] / f"{key}{suffix}"


def save_npy(path: Path, array: np.ndarray) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "wb") as handle:
            np.save(handle, array)

    atomic_write(path, write)


def fp_cache_config() -> dict:
    return {
        "rdkit_plus": RXN["rdkit_plus"],
        "drfp": RXN["drfp"],
        "standardize": RXN["standardize"],
        "dtype": "float32",
    }


def protein_cache_config(smoke: bool) -> dict:
    cfg = {
        k: PROT[k]
        for k in (
            "model",
            "revision",
            "dim",
            "pooling",
            "chunk_length",
            "tokenizer",
            "dtype",
            "batch_size",
        )
    }
    return {**cfg, "smoke_random_vectors": smoke}


# ---------------------------------------------------------------------------------------------------------
# Reaction fingerprints (original Horizyn classes)
# ---------------------------------------------------------------------------------------------------------
def make_fingerprint_datasets(reaction_dataset):
    from horizyn.datasets.fingerprints import DRFPFingerprintDataset, RDKitPlusFingerprintDataset

    rdkit = RDKitPlusFingerprintDataset(
        reaction_dataset=reaction_dataset,
        standardize=RXN["standardize"]["enabled"],
        **RXN["rdkit_plus"],
        **STD,
    )
    drfp = DRFPFingerprintDataset(
        reaction_dataset=reaction_dataset,
        standardize=RXN["standardize"]["enabled"],
        **RXN["drfp"],
        **STD,
    )
    return rdkit, drfp


def fingerprint_chunk(smiles_list: list[str]) -> list[tuple[str, np.ndarray | None]]:
    """Fingerprints of the forward and reverse direction of each reaction. None marks a reaction that fails."""
    from horizyn.datasets.base import BaseDataset

    keys, data = [], []
    for smiles in smiles_list:
        rid = reaction_id_for_smiles(smiles)
        left, right = smiles.split(">>")
        keys += [f"{rid}_f", f"{rid}_r"]
        data += [{"reaction_smiles": smiles}, {"reaction_smiles": f"{right}>>{left}"}]
    rdkit, drfp = make_fingerprint_datasets(BaseDataset(keys=keys, array_data=data))
    out = []
    for smiles in smiles_list:
        rid = reaction_id_for_smiles(smiles)
        try:
            fps = [torch.cat([rdkit[f"{rid}_{d}"], drfp[f"{rid}_{d}"]]).numpy() for d in "fr"]
            out.append((rid, np.stack(fps).astype(np.float32)))
        except MemoryError:  # a failure of the machine, not of this reaction
            raise
        except Exception as exc:  # the original featurizer cannot process this reaction
            print(f"[emulator_bench] dropping {rid}: {exc}", flush=True)
            out.append((rid, None))
    return out


def read_fingerprint_index(root: Path) -> tuple[dict[str, tuple[str, int]], set[str]]:
    """Map each cached reaction id to (shard name, row). The second value holds the ids the featurizer cannot process."""
    index, bad = {}, set()
    for meta in sorted(root.glob("shard_*.json")):
        info = read_json(meta)
        shard = meta.stem
        index.update({rid: (shard, row) for row, rid in enumerate(info["ids"])})
        bad.update(info["bad"])
    return index, bad


def write_fingerprint_shard(
    root: Path, ids: list[str], fps: list[np.ndarray], bad: list[str]
) -> None:
    """Write one shard: a float32 array [reactions, 2 directions, 2048] and its id list. The id file marks the shard as complete."""
    shard = f"shard_{uuid.uuid4().hex}"  # unique, so parallel processes never write the same shard
    save_npy(
        root / f"{shard}.npy",
        np.stack(fps) if fps else np.empty((0, 2, 2 * FP_DIM), np.float32),
    )
    write_json(root / f"{shard}.json", {"ids": ids, "bad": bad})


def fill_fingerprint_cache(root: Path, smiles_list: list[str], num_workers: int) -> None:
    index, bad = read_fingerprint_index(root)
    ids = {s: reaction_id_for_smiles(s) for s in smiles_list}
    missing = [s for s in smiles_list if ids[s] not in index and ids[s] not in bad]
    chunks = [missing[i : i + 256] for i in range(0, len(missing), 256)]
    good_ids, good_fps, bad_ids = [], [], []
    with mp.get_context("fork").Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(fingerprint_chunk, chunks),
            total=len(chunks),
            desc="reaction fingerprints",
        ):
            for rid, fps in result:
                if fps is None:
                    bad_ids.append(rid)
                else:
                    good_ids.append(rid)
                    good_fps.append(fps)
            if len(good_ids) >= SHARD_ROWS:
                write_fingerprint_shard(root, good_ids, good_fps, bad_ids)
                good_ids, good_fps, bad_ids = [], [], []
    if good_ids or bad_ids:
        write_fingerprint_shard(root, good_ids, good_fps, bad_ids)


def load_fingerprints(root: Path, reaction_ids: list[str]) -> np.ndarray:
    """Gather the fingerprints [reactions, 2, 2048] of the given ids from the memory-mapped shards."""
    index, _ = read_fingerprint_index(root)
    out = np.empty((len(reaction_ids), 2, 2 * FP_DIM), np.float32)
    by_shard: dict[str, tuple[list[int], list[int]]] = {}
    for position, rid in enumerate(reaction_ids):
        shard, row = index[rid]
        positions, rows = by_shard.setdefault(shard, ([], []))
        positions.append(position)
        rows.append(row)
    for shard, (positions, rows) in by_shard.items():
        out[positions] = np.load(root / f"{shard}.npy", mmap_mode="r")[rows]
    return out


# ---------------------------------------------------------------------------------------------------------
# Protein vectors
# ---------------------------------------------------------------------------------------------------------
def smoke_vector(sequence: str) -> np.ndarray:
    seed = int(sha256_text(sequence)[:8], 16)
    vec = np.random.default_rng(seed).standard_normal(PROT["dim"]).astype(np.float32)
    return vec / np.linalg.norm(vec)


def embed_with_prott5(missing: pd.DataFrame, root: Path, device: str) -> None:
    from transformers import T5EncoderModel, T5Tokenizer

    device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    tokenizer = T5Tokenizer.from_pretrained(
        PROT["model"], revision=PROT["revision"], do_lower_case=False
    )
    model = (
        T5EncoderModel.from_pretrained(
            PROT["model"],
            revision=PROT["revision"],
            use_safetensors=True,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )
        .to(device)
        .eval()
    )
    size = PROT["chunk_length"]
    chunks = [
        (r.protein_id, r.sequence[i : i + size])
        for r in missing.itertuples()
        for i in range(0, len(r.sequence), size)
    ]
    chunk_counts = Counter(pid for pid, _ in chunks)
    sums: dict[str, np.ndarray] = {}
    seen: Counter = Counter()
    for start in tqdm(range(0, len(chunks), PROT["batch_size"]), desc="ProtT5 embeddings"):
        batch = chunks[start : start + PROT["batch_size"]]
        text = [" ".join(seq.translate(str.maketrans("UZOB", "XXXX"))) for _, seq in batch]
        tokens = {
            k: v.to(device)
            for k, v in tokenizer(
                text, add_special_tokens=True, padding=True, return_tensors="pt"
            ).items()
        }
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state.float().cpu()
        for i, (pid, seq) in enumerate(batch):
            sums[pid] = sums.get(pid, 0) + hidden[i, : len(seq)].mean(dim=0).numpy()
            seen[pid] += 1
            if seen[pid] == chunk_counts[pid]:
                vec = (sums.pop(pid) / chunk_counts[pid]).astype(np.float32)
                save_npy(item_path(root, pid, ".npy"), vec)


def fill_protein_cache(
    root: Path, proteins: pd.DataFrame, source: str, device: str, shard: tuple[int, int] = (0, 1)
) -> None:
    missing = proteins[
        [not item_path(root, pid, ".npy").exists() for pid in proteins["protein_id"]]
    ]
    missing = missing.iloc[shard[0] :: shard[1]]
    print(
        f"[emulator_bench] proteins: {len(proteins)} needed, {len(missing)} to embed", flush=True
    )
    if missing.empty:
        return
    if source == "deterministic":
        for r in missing.itertuples():
            save_npy(item_path(root, r.protein_id, ".npy"), smoke_vector(r.sequence))
    else:
        embed_with_prott5(missing, root, device)


# ---------------------------------------------------------------------------------------------------------
# One split
# ---------------------------------------------------------------------------------------------------------
def unique_proteins(records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    proteins = pd.concat(
        [df[["protein_id", "sequence"]] for df in records.values()]
    ).drop_duplicates("protein_id")
    return proteins.sort_values("protein_id").reset_index(drop=True)


def prepare_split(
    split: SplitDir,
    *,
    out_root: Path,
    limit_per_split: int | None = None,
    embedding_source: str = "prott5",
    device: str = "auto",
    num_workers: int = 8,
    embed_shard: tuple[int, int] | None = None,
) -> dict | None:
    """Write <split_dir>/data/ for one split. Reactions the original featurizer cannot process are dropped and counted.

    With embed_shard=(k, n) it only embeds every n-th missing protein, starting at k, and writes nothing else.
    """
    data_dir = split_dir_of(out_root, split.group, split.name) / "data"
    fp_root = cache_dir(out_root, FP_CACHE, fp_cache_config())
    smoke = embedding_source == "deterministic"
    protein_root = cache_dir(
        out_root, SMOKE_CACHE if smoke else PROTEIN_CACHE, protein_cache_config(smoke)
    )

    records, stats = {}, {}
    for name in SPLIT_NAMES:
        records[name], stats[name] = load_records(split.path / f"{name}.parquet", limit_per_split)
    if embed_shard:
        fill_protein_cache(
            protein_root, unique_proteins(records), embedding_source, device, embed_shard
        )
        return None
    all_smiles = sorted(set().union(*(set(df["reaction_smiles"]) for df in records.values())))
    fill_fingerprint_cache(fp_root, all_smiles, num_workers)
    bad = read_fingerprint_index(fp_root)[1]

    # The candidate pool holds every protein of the split, also those of dropped reactions.
    all_records = dict(records)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_records.values())[["protein_id", "ec_number"]].drop_duplicates().to_csv(
        data_dir / "pool_manifest.csv", index=False
    )

    for name, df in records.items():
        keep = ~df["reaction_id"].isin(bad)
        stats[name]["dropped_unfeaturizable_reactions"] = int(
            df.loc[~keep, "reaction_id"].nunique()
        )
        stats[name]["dropped_unfeaturizable_pairs"] = int((~keep).sum())
        # Evaluation scores these reactions as misses.
        df[~keep].groupby(["reaction_id", "reaction_smiles"], as_index=False).agg(
            ec_number=(
                "ec_number",
                lambda v: ";".join(sorted({e for x in v for e in x.split(";") if e})),
            )
        ).to_csv(data_dir / f"{name}_dropped.csv", index=False)
        records[name] = df[keep].reset_index(drop=True)
        if records[name].empty:
            raise ValueError(f"{split.label}/{name} has no usable rows")
        stats[name].update(
            pairs=int(len(records[name])),
            reactions=int(records[name]["reaction_id"].nunique()),
            proteins=int(records[name]["protein_id"].nunique()),
            pairs_with_full_ec=int(
                records[name]["ec_number"].map(lambda v: bool(full_ec_labels(v))).sum()
            ),
        )
        pairs = records[name][["reaction_id", "protein_id"]].copy()
        pairs.insert(0, "pr_id", "pair_" + pairs.index.astype(str))
        data_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(data_dir / f"{name}_pairs.csv", index=False)
        records[name][["reaction_id", "reaction_smiles"]].drop_duplicates().sort_values(
            "reaction_id"
        ).to_csv(data_dir / f"{name}_rxns.csv", index=False)
        records[name].drop(columns="sequence").to_csv(
            data_dir / f"{name}_manifest.csv", index=False
        )

    proteins = unique_proteins(all_records)
    fill_protein_cache(protein_root, proteins, embedding_source, device)
    vectors = np.stack(
        [
            np.load(item_path(protein_root, pid, ".npy"))
            for pid in tqdm(proteins["protein_id"], desc="protein HDF5")
        ]
    )

    def write_h5(tmp: str) -> None:
        with h5py.File(tmp, "w") as handle:
            handle.create_dataset(
                "ids",
                data=np.array(proteins["protein_id"], dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            handle.create_dataset("vectors", data=vectors.astype(np.float32))

    atomic_write(data_dir / "prots_t5.h5", write_h5)

    rxn_ids = sorted(set().union(*(set(df["reaction_id"]) for df in records.values())))
    id_to_smiles = {reaction_id_for_smiles(s): s for s in all_smiles}
    fps = load_fingerprints(fp_root, rxn_ids)

    def write_fps(tmp: str) -> None:
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                ids=np.array(rxn_ids),
                fps=fps,
                smiles=np.array([id_to_smiles[i] for i in rxn_ids]),
            )

    atomic_write(data_dir / "rxn_fps.npz", write_fps)
    summary = {
        "split": split.label,
        "limit_per_split": limit_per_split,
        "embedding_source": embedding_source,
        "stats": stats,
        "unique_proteins": int(len(proteins)),
    }
    write_json(data_dir / "data_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Split labels (group/name) or groups. Default: all default splits found on disk.",
    )
    parser.add_argument(
        "--limit_per_split",
        type=int,
        default=None,
        help="Smoke test only: random rows per split file.",
    )
    parser.add_argument(
        "--embedding_source",
        choices=["prott5", "deterministic"],
        default="prott5",
        help="'deterministic' writes random vectors to a separate smoke cache. Smoke test only.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--embed_shard",
        help="K/N: only embed every N-th missing protein, starting at K, then stop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, skipped = discover_splits(args.dataset_root, args.splits)
    for line in skipped:
        print(f"[emulator_bench] skipped {line}", flush=True)
    shard = tuple(map(int, args.embed_shard.split("/"))) if args.embed_shard else None
    for split in splits:
        summary = prepare_split(
            split,
            out_root=Path(args.out_root),
            limit_per_split=args.limit_per_split,
            embedding_source=args.embedding_source,
            device=args.device,
            num_workers=args.num_workers,
            embed_shard=shard,
        )
        if shard:
            continue
        print(f"[emulator_bench] prepared {split.label}: {summary['stats']}", flush=True)


if __name__ == "__main__":
    main()
