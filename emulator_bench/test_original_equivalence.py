"""The wrapper computes what the original Horizyn code computes.

Compared on tiny CPU inputs: reaction fingerprints, the pair batches, one training step (loss and updated
weights) and the retrieval ranking. Tensors are float64 where the code allows it, so equal results are
bit-equal or equal to rounding noise.
"""

import importlib.util
import math

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from horizyn import data_module
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset, TupleDataset
from horizyn.datasets.fingerprints import DRFPFingerprintDataset, RDKitPlusFingerprintDataset
from horizyn.datasets.fingerprints.base import BaseFingerprintDataset
from horizyn.datasets.transform import ConcatTensorTransform
from horizyn.lightning_module import HorizynLitModule
from torch.utils.data import DataLoader

from emulator_bench import cache_features, evaluate, fit
from emulator_bench.utils import BASELINE_ROOT, reaction_id_for_smiles

REACTIONS = ["CC(=O)O.OCC>>CC(=O)OCC.O", "CCO>>CC=O", "OC(=O)CC(=O)C(O)=O>>CC(=O)C(O)=O.O=C=O"]


def original_query_dataset(reactions: dict[str, str]) -> MergeDataset:
    """The reaction fingerprint dataset as horizyn.data_module builds it (sota.yaml settings)."""
    keys, data = [], []
    for rid, smiles in reactions.items():
        left, right = smiles.split(">>")
        keys += [f"{rid}_f", f"{rid}_r"]
        data += [{"reaction_smiles": smiles}, {"reaction_smiles": f"{right}>>{left}"}]
    base = BaseDataset(keys=keys, array_data=data)
    standardize = dict(
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )
    rdkit = RDKitPlusFingerprintDataset(
        reaction_dataset=base,
        vec_dim=1024,
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        **standardize,
    )
    drfp = DRFPFingerprintDataset(
        reaction_dataset=base, vec_dim=1024, radius=3, rings=True, **standardize
    )
    merged = MergeDataset(datasets={"rdkit": rdkit, "drfp": drfp}, add_prefix=False)
    merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))
    return merged


def test_cached_fingerprints_equal_original_fingerprints(tmp_path, monkeypatch):
    ids = [reaction_id_for_smiles(s) for s in REACTIONS]
    reactions = dict(zip(ids, REACTIONS))
    reference = original_query_dataset(reactions)
    expected = {key: reference[key] for key in reference.keys}

    chunk = cache_features.fingerprint_chunk(REACTIONS)
    assert [rid for rid, _ in chunk] == ids and all(fps is not None for _, fps in chunk)
    path = tmp_path / "rxn_fps.npz"
    np.savez(
        path, ids=np.array(ids), smiles=np.array(REACTIONS), fps=np.stack([f for _, f in chunk])
    )

    def refuse(self, reaction_info):
        raise AssertionError("fingerprint was computed, not read from the cache")

    monkeypatch.setattr(BaseFingerprintDataset, "__init__", BaseFingerprintDataset.__init__)
    fit.install_fingerprint_preload([str(path)])
    monkeypatch.setattr(RDKitPlusFingerprintDataset, "_generate_fingerprint", refuse)
    monkeypatch.setattr(DRFPFingerprintDataset, "_generate_fingerprint", refuse)
    preloaded = original_query_dataset(reactions)
    for key in expected:
        # Cached rows are float32 copies of the original float32 fingerprints: bit-equal.
        assert torch.equal(preloaded[key], expected[key])
        assert preloaded[key].shape == (2048,)


def make_pairs(dtype=torch.float64) -> TupleDataset:
    gen = lambda seed: torch.Generator().manual_seed(seed)  # noqa: E731
    queries = BaseDataset(
        keys=[f"q{i}" for i in range(9)],
        array_data=torch.randn(9, 6, generator=gen(1), dtype=dtype),
    )
    targets = BaseDataset(
        keys=[f"t{i}" for i in range(7)],
        array_data=torch.randn(7, 5, generator=gen(2), dtype=dtype),
    )
    pairs = BaseDataset(
        keys=[f"p{i}" for i in range(24)],
        array_data=[{"query_id": f"q{i % 9}", "target_id": f"t{i * 5 % 7}"} for i in range(24)],
    )
    return TupleDataset(
        pairs,
        {"query_id": queries, "target_id": targets},
        rename_map={"query_id": "query_vec", "target_id": "target_vec"},
    )


def loader_batches(dataset: TupleDataset) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(4),
        collate_fn=data_module.default_collate,
    )
    return [{k: v.cpu() if torch.is_tensor(v) else v for k, v in b.items()} for b in loader]


@pytest.fixture
def original_and_patched_batches(monkeypatch):
    """Batches of the original TupleDataset and of the fit.py patched one (same shuffle seed)."""
    monkeypatch.setattr(fit, "DEVICE", "cpu")
    for target, name in (
        (TupleDataset, "_filter_valid_keys"),
        (TupleDataset, "__getitems__"),
        (data_module, "default_collate"),
    ):
        monkeypatch.setattr(target, name, getattr(target, name, None), raising=False)
    original = loader_batches(make_pairs())
    fit.install_batched_pairs()
    return original, loader_batches(make_pairs())


def lit_module() -> HorizynLitModule:
    torch.manual_seed(0)
    return HorizynLitModule(
        query_encoder_dims=[6, 12, 12, 8],
        target_encoder_dims=[5, 12, 12, 8],
        embedding_dim=8,
        learning_rate=1e-2,
    ).double()


def train_one_epoch(batches: list[dict]) -> tuple[HorizynLitModule, list[float]]:
    losses = []

    class Recorder(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            losses.append(outputs["loss"].item())

    module = lit_module()
    pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[Recorder()],
        default_root_dir=None,
    ).fit(module, DataLoader(batches, batch_size=None))
    return module, losses


def test_training_steps_on_patched_batches_equal_original(original_and_patched_batches):
    original, patched = original_and_patched_batches
    assert len(original) == len(patched) == 3
    expected, expected_losses = train_one_epoch(original)
    got, got_losses = train_one_epoch(patched)
    # Identical float64 batches go through identical code: losses and weights are bit-equal.
    assert got_losses == expected_losses
    for (name, a), (_, b) in zip(got.state_dict().items(), expected.state_dict().items()):
        assert torch.equal(a, b), name


def test_training_loss_and_adamw_step_match_hand_computation(original_and_patched_batches):
    batch = original_and_patched_batches[1][0]
    module = lit_module()
    initial = {k: v.clone() for k, v in module.state_dict().items()}

    # Reference: unique ids sorted as np.unique does, loss = beta * mean(d_pos) + logsumexp(-beta * d).
    q_ids, t_ids = batch["query_id"], batch["target_id"]
    uq, qi = np.unique(q_ids, return_inverse=True)
    ut, ti = np.unique(t_ids, return_inverse=True)
    q_first = [q_ids.index(i) for i in uq]
    t_first = [t_ids.index(i) for i in ut]
    q, t = module.model(batch["query_vec"][q_first], batch["target_vec"][t_first])
    dists = 1 - q @ t.T
    beta = math.exp(
        torch.log(torch.tensor(10.0)).item()
    )  # the original stores log(beta), rounded to float32
    expected = beta * dists[torch.as_tensor(qi), torch.as_tensor(ti)].mean() + torch.logsumexp(
        -beta * dists.flatten(), 0
    )

    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    expected_grads = torch.autograd.grad(expected, params)
    optimizer = module.configure_optimizers()
    loss = module.training_step(batch, 0)
    assert torch.allclose(
        loss, expected, rtol=0, atol=1e-12
    )  # float64 summation order only  # float64 summation order only
    loss.backward()
    optimizer.step()
    for (name, p), grad in zip(named, expected_grads):
        assert torch.allclose(p.grad, grad, rtol=1e-10, atol=1e-13), name
        # First AdamW step: decoupled decay, then lr * g / (|g| + eps) since m_hat = g and v_hat = g^2.
        step = 1e-2 * grad / (grad.abs() + 1e-8)
        assert torch.allclose(
            p.detach(), initial[name] * (1 - 1e-2 * 0.01) - step, rtol=0, atol=1e-10
        ), name


def test_ranked_ecs_equal_original_scoring():
    spec = importlib.util.spec_from_file_location(
        "horizyn_native_evaluate", BASELINE_ROOT / "scripts" / "evaluate.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    module = lit_module().float().eval()
    gen = np.random.default_rng(5)
    n_rxn, n_prot = 11, 14
    query_fps = gen.standard_normal((n_rxn, 2, 6)).astype(np.float32)
    protein_vecs = torch.from_numpy(gen.standard_normal((n_prot, 5)).astype(np.float32))
    ec_pairs = [(p, p % 4) for p in range(n_prot)] + [(0, 3), (5, 1)]  # a protein can hold two ECs
    num_ecs = 4
    with torch.no_grad():
        target_embeds = module.model.target_encoder(protein_vecs)
        got = evaluate.rank_ecs(
            module,
            query_fps,
            target_embeds,
            torch.tensor([p for p, _ in ec_pairs]),
            torch.tensor([e for _, e in ec_pairs]),
            num_ecs,
            batch=4,
        )

        expected = []
        for fps in query_fps:
            # Original per-query scoring: scores = -cosine distance, one row per direction.
            scores = torch.stack(
                [
                    -native.compute_cosine_distances(
                        module.model.query_encoder(torch.from_numpy(f)[None]), target_embeds
                    ).squeeze(0)
                    for f in fps
                ]
            ).amax(0)
            best = torch.full((num_ecs,), -torch.inf, dtype=torch.float32)
            for p, e in ec_pairs:
                best[e] = max(best[e], scores[p])
            expected.append(best.argsort(descending=True).tolist())
    # float32 matmuls of different shapes differ by rounding noise, far below the score gaps.
    assert got == [row[: len(got[0])] for row in expected]
