"""The fit.py patches give the same pairs and batches as the original TupleDataset and DataLoader."""

import torch
from horizyn import data_module
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import TupleDataset
from torch.utils.data import DataLoader

from emulator_bench import fit


def make_pairs() -> TupleDataset:
    queries = BaseDataset(
        keys=[f"q{i}" for i in range(20)],
        array_data=torch.randn(20, 8, generator=torch.Generator().manual_seed(1)),
    )
    targets = BaseDataset(
        keys=[f"t{i}" for i in range(15)],
        array_data=torch.randn(15, 4, generator=torch.Generator().manual_seed(2)),
    )
    pairs = BaseDataset(
        keys=[f"p{i}" for i in range(100)],
        array_data=[
            {"query_id": f"q{i % 23}", "target_id": f"t{i * 7 % 15}"} for i in range(100)
        ],  # q20 to q22 do not exist
    )
    return TupleDataset(
        pairs,
        {"query_id": queries, "target_id": targets},
        rename_map={"query_id": "query_vec", "target_id": "target_vec"},
    )


def batches(dataset: TupleDataset, collate) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(3),
        collate_fn=collate,
    )
    return [
        {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()} for batch in loader
    ]


def test_batched_pairs_match_original(monkeypatch):
    expected_keys = make_pairs().keys
    expected = batches(make_pairs(), data_module.default_collate)
    for target, name in (
        (TupleDataset, "_filter_valid_keys"),
        (TupleDataset, "__getitems__"),
        (data_module, "default_collate"),
    ):
        monkeypatch.setattr(target, name, getattr(target, name, None), raising=False)
    fit.install_batched_pairs()
    patched = make_pairs()
    assert patched.keys == expected_keys and len(expected_keys) < 100
    got = batches(patched, data_module.default_collate)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert list(a) == list(b)
        assert a["query_id"] == b["query_id"] and a["target_id"] == b["target_id"]
        assert torch.equal(a["query_vec"], b["query_vec"]) and torch.equal(
            a["target_vec"], b["target_vec"]
        )


def test_last_checkpoint_reaches_final_epoch(tmp_path, monkeypatch):
    import lightning.pytorch as pl

    from emulator_bench import train

    monkeypatch.setattr(pl.callbacks, "ModelCheckpoint", pl.callbacks.ModelCheckpoint)
    fit.install_last_every_epoch()

    class Model(pl.LightningModule):
        def __init__(self, width: int = 2):
            super().__init__()
            self.save_hyperparameters()
            self.layer = torch.nn.Linear(width, 1)

        def training_step(self, batch, batch_idx):
            return self.layer(batch[0]).pow(2).mean()

        def validation_step(self, batch, batch_idx):
            self.log("val/loss", float(self.current_epoch))  # worse after the first validation

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    data = DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=4)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path / "lightning", every_n_epochs=2, save_last=True, save_top_k=3,
        monitor="val/loss", mode="min",
    )
    pl.Trainer(
        max_epochs=7, callbacks=[checkpoint], check_val_every_n_epoch=2, logger=False,
        enable_progress_bar=False, accelerator="cpu", default_root_dir=tmp_path,
    ).fit(Model(), data, data)
    assert torch.load(tmp_path / "lightning" / "last.ckpt", weights_only=False)["epoch"] == 6
    assert train.export_checkpoints(tmp_path, 7)["last_epoch"] == 6
