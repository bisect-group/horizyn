import os
import runpy
import sys

import lightning.pytorch as pl
import pytest
import torch
import yaml

from emulator_bench import fit
from emulator_bench.train import build_config
from emulator_bench.utils import BASELINE_ROOT

K = 10  # the production gap must stay within 10x the bf16-path noise (perturbed weights) gap
NOISE_U = 2**-8  # one rounding of the weights at the production precision (bf16 unit roundoff)
EPOCHS = 4
NEGATIVE_CONTROL = os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1"


def train(data_dir, run_dir, monkeypatch, precision: str, noise: bool = False, scale: float = 1.0):
    """Run the production entry (`emulator_bench.fit` on top of the original train.py) and return the step losses."""
    losses, final = [], {}

    class Recorder(pl.Callback):
        def on_fit_start(self, trainer, module):
            generator = torch.Generator().manual_seed(123)
            with torch.no_grad():
                for p in module.parameters():
                    if noise:
                        p += NOISE_U * p.abs() * torch.randn(p.shape, generator=generator).to(p.device)
                    p *= scale

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            losses.append(outputs["loss"].item())

        def on_fit_end(self, trainer, module):
            final["val"] = trainer.callback_metrics["val/loss"].item()

    config = build_config(data_dir, run_dir, 666, EPOCHS, precision)
    config["model"].update(
        query_encoder_dims=[2048, 32, 32, 16], target_encoder_dims=[1024, 32, 32, 16], embedding_dim=16
    )
    config["data"]["train_batch_size"] = 16
    config["training"]["check_val_every_n_epoch"] = EPOCHS
    run_dir.mkdir(parents=True)
    config_path = run_dir.with_suffix(".yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    original_trainer_init = pl.Trainer.__init__

    def trainer_init(self, *args, **kwargs):
        original_trainer_init(self, *args, **{**kwargs, "callbacks": [*kwargs["callbacks"], Recorder()]})

    with monkeypatch.context() as m:
        m.setattr(pl.Trainer, "__init__", trainer_init)
        m.setattr(sys, "argv", ["train.py", "--config", str(config_path), "--seed", "666"])
        try:
            runpy.run_path(str(BASELINE_ROOT / "train.py"), run_name="__main__")
        except SystemExit as exit_:
            assert exit_.code in (None, 0), "train.py failed"
    return losses, final["val"]


@pytest.fixture
def production_patches(monkeypatch, fixture_data_dir):
    """Install the wrapper patches of `emulator_bench.fit` (restored after the test)."""
    from horizyn import data_module
    from horizyn.datasets.collection import TupleDataset
    from horizyn.datasets.fingerprints.base import BaseFingerprintDataset

    for owner, name in (
        (BaseFingerprintDataset, "__init__"),
        (TupleDataset, "_filter_valid_keys"),
        (TupleDataset, "__getitems__"),
        (data_module, "default_collate"),
        (pl.callbacks, "ModelCheckpoint"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name, None), raising=False)
    fit.install_fingerprint_preload([str(fixture_data_dir / "rxn_fps.npz")])
    fit.install_batched_pairs()
    fit.install_last_every_epoch()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bf16_training_within_fp32_noise_control(production_patches, fixture_data_dir, tmp_path, monkeypatch):
    """Production precision (bf16 autocast, `--amp`) against the original fp32, both through `emulator_bench.fit`.

    The run is the production one: the on-GPU batch gather of `fit.py`, `deterministic=True` as in the original
    train.py. Same seed, same batches. A noise control reruns the bf16 path with the initial weights perturbed by 2**-8
    (relative, one bf16 rounding). Floor = |perturbed bf16 - bf16|, gap = |bf16 - fp32|. The per-step loss gap and the
    final validation loss gap must stay within K times the floor. Step 0 is judged like every other step.
    With EQUIV_NEGATIVE_CONTROL=1 the production run starts from weights multiplied by 1.05; the test must then fail.
    """
    ref, ref_val = train(fixture_data_dir, tmp_path / "ref", monkeypatch, "32-true")
    prod, prod_val = train(
        fixture_data_dir, tmp_path / "prod", monkeypatch, "bf16-mixed", scale=1.05 if NEGATIVE_CONTROL else 1.0
    )
    noise, noise_val = train(fixture_data_dir, tmp_path / "noise", monkeypatch, "bf16-mixed", noise=True)
    ref, noise, prod = (torch.tensor(x, dtype=torch.float64) for x in (ref, noise, prod))
    assert len(ref) == len(noise) == len(prod) > 0
    tiny = 1e-7 * ref.abs().max()
    gap, floor = (prod - ref).abs(), (noise - prod).abs()
    final_gap, final_floor = abs(prod_val - ref_val), abs(noise_val - prod_val)
    final_tiny = 1e-7 * abs(ref_val)
    tiny0 = 1e-7 * ref[0].abs()
    gap0, floor0 = float(gap[0]), float(floor[0])
    print(
        f"horizyn bf16: max gap={gap.max():.3e}, max floor={floor.max():.3e}, "
        f"ratio={gap.max() / max(floor.max(), tiny):.2f}, K={K}, step0 gap={gap0:.3e}, floor={floor0:.3e}, "
        f"ratio={gap0 / max(floor0, tiny0):.2f}, final gap={final_gap:.3e}, final floor={final_floor:.3e}, "
        f"final ratio={final_gap / max(final_floor, final_tiny):.2f}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap.max() <= K * max(floor.max(), tiny)
    assert final_gap <= K * max(final_floor, final_tiny)
