"""The wrapper training entry gives the same weights and metrics as the original train.py.

Both run a few epochs on the synthetic fixture split, on CPU in fp32, with the same seed and thread count.
The model is smaller than the published one (hidden size 32). The ProtT5 vectors are deterministic stand-ins.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest
import torch
import yaml

from emulator_bench.train import build_config
from emulator_bench.utils import BASELINE_ROOT

EPOCHS = 3


def run_training(entry: list[str], data_dir, run_dir, extra: list[str] = ()) -> None:
    config = build_config(data_dir, run_dir, 666, EPOCHS, "32-true")
    config["model"].update(
        query_encoder_dims=[2048, 32, 32, 16], target_encoder_dims=[1024, 32, 32, 16], embedding_dim=16
    )
    config["data"]["train_batch_size"] = 16
    config["training"]["check_val_every_n_epoch"] = 1
    config_path = run_dir.with_suffix(".yaml")
    run_dir.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    command = [sys.executable, *entry, *extra, "--config", str(config_path), "--seed", "666"]
    subprocess.run(command, cwd=BASELINE_ROOT, env=env, check=True, capture_output=True)


def test_wrapper_training_equals_original_train_py(fixture_data_dir, tmp_path):
    run_training(["train.py"], fixture_data_dir, tmp_path / "original")
    run_training(
        ["-m", "emulator_bench.fit"],
        fixture_data_dir,
        tmp_path / "wrapper",
        ["--fp_file", str(fixture_data_dir / "rxn_fps.npz"), "--"],
    )
    ckpts = {
        name: torch.load(tmp_path / name / "lightning" / "last.ckpt", map_location="cpu", weights_only=False)
        for name in ("original", "wrapper")
    }
    assert ckpts["original"]["epoch"] == ckpts["wrapper"]["epoch"] == EPOCHS - 1
    # Same batches, same code, same threads: the weights are bit-equal.
    for name, tensor in ckpts["original"]["state_dict"].items():
        assert torch.equal(tensor, ckpts["wrapper"]["state_dict"][name]), name

    logs = {
        name: pd.read_csv(next((tmp_path / name / "logs").rglob("metrics.csv")))
        for name in ("original", "wrapper")
    }
    assert "val/loss" in logs["original"].columns
    pd.testing.assert_frame_equal(logs["original"], logs["wrapper"], check_exact=True)
