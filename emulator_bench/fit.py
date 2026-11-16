"""Run the original train.py with cached reaction fingerprints.

The original fingerprint classes compute a fingerprint on first use and keep it in memory. This module fills
that memory from the fingerprints computed by cache_features. The values are the same. Everything else in
train.py runs unchanged.

Usage: python -m emulator_bench.fit --fp_file <rxn_fps.npz> [...] -- --config <yaml> --seed <n> [--resume <ckpt>]
"""

from __future__ import annotations

import argparse
import runpy
import sys

import numpy as np
import torch

from .utils import BASELINE_ROOT, HPARAMS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RDKIT_DIM = HPARAMS["reaction_features"]["rdkit_plus"]["vec_dim"]
DRFP_DIM = HPARAMS["reaction_features"]["drfp"]["vec_dim"]


def install_fingerprint_preload(fp_files: list[str]) -> None:
    from horizyn.datasets.fingerprints import RDKitPlusFingerprintDataset
    from horizyn.datasets.fingerprints.base import BaseFingerprintDataset

    store = {}
    for path in fp_files:
        with np.load(path) as z:
            store.update(
                {i: (s, torch.from_numpy(f)) for i, s, f in zip(z["ids"], z["smiles"], z["fps"])}
            )
    original_init = BaseFingerprintDataset.__init__

    def init(self, reaction_dataset, *args, **kwargs):
        original_init(self, reaction_dataset, *args, **kwargs)
        is_rdkit = isinstance(self, RDKitPlusFingerprintDataset)
        if self.vec_dim != (RDKIT_DIM if is_rdkit else DRFP_DIM):
            return
        columns = slice(0, RDKIT_DIM) if is_rdkit else slice(RDKIT_DIM, RDKIT_DIM + DRFP_DIM)
        for key in reaction_dataset.keys:
            rid, direction = key.rsplit("_", 1)
            if rid not in store:
                continue
            smiles, fps = store[rid]
            left, right = smiles.split(">>")
            expected = smiles if direction == "f" else f"{right}>>{left}"
            if reaction_dataset[key]["reaction_smiles"] == expected:
                self._cache.set(key, fps[0 if direction == "f" else 1, columns].clone())

    BaseFingerprintDataset.__init__ = init


def valid_pair_keys(self) -> list:
    """Same result as TupleDataset._filter_valid_keys, with set lookups instead of list scans."""
    key_sets = {name: set(dataset.keys) for name, dataset in self.key_name_to_dataset.items()}
    valid = []
    for key in self.tuple_dataset.keys:
        pair = self.tuple_dataset[key]
        if all(pair[name] in keys for name, keys in key_sets.items()):
            valid.append(key)
    return valid


def batch_of_pairs(self, indices: list[int]) -> dict:
    """Return the collated batch of the given samples.

    Each distinct id is looked up once. The vectors stay on the GPU and the batch rows are gathered there.
    """
    if not hasattr(self, "_tables"):
        pairs = [self.tuple_dataset[key] for key in self.keys]
        self._tables = {}
        for name, dataset in self.key_name_to_dataset.items():
            ids = [pair[name] for pair in pairs]
            unique = list(dict.fromkeys(ids))
            position = {uid: i for i, uid in enumerate(unique)}
            rows = torch.from_numpy(
                np.fromiter((position[i] for i in ids), dtype=np.int64, count=len(ids))
            )
            table = torch.stack([dataset[uid] for uid in unique])
            self._tables[name] = (np.array(ids, dtype=object), rows.to(DEVICE), table.to(DEVICE))
    index = torch.as_tensor(indices, device=DEVICE)
    batch = {name: ids[indices].tolist() for name, (ids, _, _) in self._tables.items()}
    for name, (_, rows, table) in self._tables.items():
        batch[self.rename_map.get(name, name)] = table[rows[index]]
    return batch


def install_batched_pairs() -> None:
    """Build train and validation batches in the main process (the DataLoader draws the same shuffle order)."""
    from horizyn import data_module
    from horizyn.datasets.collection import TupleDataset

    torch_collate = data_module.default_collate
    TupleDataset._filter_valid_keys = valid_pair_keys
    TupleDataset.__getitems__ = batch_of_pairs
    data_module.default_collate = (
        lambda batch: batch if isinstance(batch, dict) else torch_collate(batch)
    )


def install_last_every_epoch() -> None:
    """Write last.ckpt at every epoch end, so it holds the final epoch and a resume loses at most one epoch.

    Lightning 2.6 writes last.ckpt only when a top-k checkpoint is saved. The top-k files (best 3 val/loss) are unchanged.
    """
    import lightning.pytorch as pl

    class EveryEpochLastCheckpoint(pl.callbacks.ModelCheckpoint):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            super().on_train_epoch_end(trainer, pl_module)
            if self._last_global_step_saved != trainer.global_step:
                self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    pl.callbacks.ModelCheckpoint = EveryEpochLastCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp_file", action="append", default=[])
    args, train_args = parser.parse_known_args()
    install_fingerprint_preload(args.fp_file)
    install_batched_pairs()
    install_last_every_epoch()
    sys.argv = ["train.py", *[a for a in train_args if a != "--"]]
    runpy.run_path(str(BASELINE_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
