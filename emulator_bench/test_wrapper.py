"""Checkpoint export and fingerprint cache rules of the wrapper."""

import numpy as np
import pandas as pd
import pytest
import torch

from emulator_bench import cache_features, dataset_adapter, evaluate, results, train


def write_checkpoint(path, epoch: int) -> None:
    torch.save(
        {
            "state_dict": {},
            "hyper_parameters": {},
            "pytorch-lightning_version": "x",
            "epoch": epoch,
            "global_step": 1,
        },
        path,
    )


def test_export_rejects_interrupted_fit(tmp_path):
    write_checkpoint(tmp_path / "checkpoint_last.pt", epoch=2)
    with pytest.raises(RuntimeError, match="stopped at epoch 3 of 5"):
        train.export_checkpoints(tmp_path, 5)
    assert not (tmp_path / "lastmodel.pth").exists()
    assert train.export_checkpoints(tmp_path, 3) == {"last_epoch": 2, "best_epoch": 2}
    assert (tmp_path / "bestmodel.pth").exists() and (tmp_path / "lastmodel.pth").exists()


def test_shards_have_unique_names_and_are_read_back(tmp_path):
    fp = lambda v: np.full((2, 2 * cache_features.FP_DIM), v, np.float32)  # noqa: E731
    cache_features.write_fingerprint_shard(tmp_path, ["a"], [fp(1)], ["x"])
    cache_features.write_fingerprint_shard(tmp_path, ["b"], [fp(2)], [])
    assert len(list(tmp_path.glob("shard_*.json"))) == 2
    index, bad = cache_features.read_fingerprint_index(tmp_path)
    assert bad == {"x"} and set(index) == {"a", "b"}
    assert cache_features.load_fingerprints(tmp_path, ["b", "a"])[:, 0, 0].tolist() == [2, 1]


def test_memory_error_is_not_cached_as_bad_reaction(monkeypatch):
    class Failing:
        def __getitem__(self, key):
            raise MemoryError

    monkeypatch.setattr(
        cache_features, "make_fingerprint_datasets", lambda _: (Failing(), Failing())
    )
    with pytest.raises(MemoryError):
        cache_features.fingerprint_chunk(["CC>>CCO"])


def write_split(root, group, name="all", with_test=True):
    folder = root / group / name if name != "all" else root / group
    folder.mkdir(parents=True)
    cols = {"rxn_smiles": ["A>>B"], "ec_number": ["1.1.1.1"], "sequence": ["MK"]}
    for split in ("train", "val", "test"):
        pd.DataFrame(cols).to_parquet(folder / f"{split}.parquet")


def test_discover_splits_skips_backups_and_takes_drfp(tmp_path):
    base = "with_protein_information"
    write_split(tmp_path, f"{base}/random_splits_grouped_rxn_key")
    write_split(tmp_path, f"{base}/reaction_drfp_tanimoto_splits", "threshold_0.425")
    write_split(tmp_path, f"{base}/enzyme_structure_splits/backup_2026/threshold_0.7", "all")
    labels = {s.label for s in dataset_adapter.discover_splits(tmp_path)[0]}
    assert labels == {
        f"{base}/random_splits_grouped_rxn_key/all",
        f"{base}/reaction_drfp_tanimoto_splits/threshold_0.425",
    }
    found = dataset_adapter.discover_splits(tmp_path, ["reaction_drfp_tanimoto_splits"])[0]
    assert [s.name for s in found] == ["threshold_0.425"]
    found = dataset_adapter.discover_splits(tmp_path, ["random_splits_grouped_rxn_key"])[0]
    assert [s.name for s in found] == ["all"]


def test_pool_keeps_proteins_of_dropped_reactions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rebuild"):
        evaluate.load_pool(tmp_path)
    # p3 occurs only in a reaction dropped as unfeaturizable, so it is not in any manifest.
    pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p1", "p3"],
            "ec_number": ["1.1.1.1", "2.1.1.1;3.-.-.-", "1.1.1.2", "4.1.1.1"],
        }
    ).to_csv(tmp_path / "pool_manifest.csv", index=False)
    pool = evaluate.load_pool(tmp_path)
    assert pool == {"p1": {"1.1.1.1", "1.1.1.2"}, "p2": {"2.1.1.1"}, "p3": {"4.1.1.1"}}
    ec_protein_idx = torch.tensor([0, 1, 2])
    ec_idx = torch.tensor([0, 1, 2])  # ECs 2.1.1.1, 4.1.1.1 and 1.1.1.1 per protein row

    class Model:
        class model:  # noqa: N801
            query_encoder = staticmethod(lambda x: x)

    fps = np.zeros((1, 2, 3), np.float32)
    fps[0, 0] = [0, 0, 1]
    ranked = evaluate.rank_ecs(Model, fps, torch.eye(3), ec_protein_idx, ec_idx, 3)
    assert ranked[0][0] == 2  # protein 3 scores best; its EC is rankable


def test_dropped_reactions_count_as_misses(tmp_path):
    dropped = pd.DataFrame(
        {"reaction_id": ["r1"], "reaction_smiles": ["A>>B"], "ec_number": ["1.1.1.1"]}
    )
    dropped.to_csv(tmp_path / "test_dropped.csv", index=False)
    predictions = pd.DataFrame(
        {"reaction_id": ["r0"], "Reaction": ["C>>D"], "EC number": ["1.1.1.1"], "0": ["1.1.1.1"]}
    )
    with_miss = evaluate.add_misses(predictions, tmp_path, "test")
    assert len(with_miss) == 2
    assert results.compute_care_task2_metrics(with_miss)["k=1"]["level_4_accuracy"] == 50.0
