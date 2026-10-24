# Horizyn wrapper report

## 1. Method summary

Horizyn is a dual-encoder retrieval model. A query MLP encodes a reaction (RDKit+ and DRFP fingerprints,
2048 values). A target MLP encodes a protein (mean-pooled ProtT5-XL vector, 1024 values). Both outputs are
L2-normalised. A full-batch multi-label NCE loss (beta 10) trains the pair. Reactions are used in forward and
reverse direction. EMULaToR ranks EC numbers: a reaction scores an EC by the best score of its train proteins.

## 2. Original defaults

Code wins over the paper. Full list with sources: `default_hparams.json`.

| Item | Value | Source |
|---|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 0.01, default betas, no scheduler | `configs/sota.yaml`, `horizyn/lightning_module.py` |
| Paper value | lr 5e-5, last batch padded to 16384 | paper p. 9-10 (not in code) |
| Epochs | 100, no early stopping | `configs/sota.yaml` |
| Batch | 16384 train, 128 retrieval | `configs/sota.yaml` |
| Loss | FullBatchMLNCELoss, beta 10 fixed | `configs/sota.yaml` |
| Model | query [2048,4096,4096,512], target [1024,4096,4096,512], ReLU | `configs/sota.yaml` |
| Reaction features | RDKit+ 1024 + DRFP 1024 (radius 3, rings), standardised | `configs/sota.yaml` |
| Protein features | ProtT5-XL half UniRef50 encoder, mean pool, 5000-residue chunks averaged | paper, `horizyn-user-manual.md` |
| Validation and checkpoint | every 10 epochs | `configs/sota.yaml` |
| Precision | 32-true | `train.py` default |
| Evaluated model | last epoch | original `README.md` |

## 3. What the wrapper does

1. `run_benchmarks` finds splits on disk and builds inputs once per split (`cache_features`).
2. `cache_features` dedups pairs, builds fingerprints with the original classes, embeds proteins, drops
   reactions that cannot be featurised and counts them.
3. `train` runs the original `train.py` through `fit` (fingerprints preloaded from cache, nothing else changed).
4. `evaluate` computes native metrics with the original `scripts/evaluate.py` and EC accuracy (levels 4 to 1, k up to 50).
5. `aggregate_results` rebuilds the summaries from the saved predictions.

Run folder: `OUT/enzyme_retrieval/<split_group>/<split_name>/seed_<s>/`. Flat groups use `all`.
Seeds 666, 777, 888. Launcher flags: `--gpus`, `--runs_per_gpu`, `--amp` (default off, fp32).

## 4. Files changed

Changed: `utils.py`, `dataset_adapter.py`, `cache_features.py`, `train.py`, `evaluate.py`, `results.py`,
`aggregate_results.py`, `README.md`, `../queue_multiple_seeds.sh`.
New: `default_hparams.json`, `fit.py`, `run_benchmarks.py`, `commands.txt`, `REPORT.md`, `test_fit_patches.py`.
Removed: `queue_pipeline.py` (replaced by `run_benchmarks.py`).

## 5. Bugs fixed

- The evaluation seed was hard-coded to 42. It now follows the run seed.
- The default precision was bf16. It is now fp32, the original. `--amp` is opt-in.
- Output layout did not match the spec (`runs/<slug>/seeds/...`). It now uses the spec layout and checkpoint names.
- No cache versioning. Caches now live in `embeddings/<kind>_v<N>/` with a verified `config.json`.
- Splits without a `sequence` column were not rejected clearly. They are now skipped with a message.
- Extra evaluation options (centroid, cosine, mean, supplemental metrics) are removed. They are not the default protocol.
- Truth labels now use four-level EC numbers only.
- `num_workers` was 20. It is now 0 because batches are built in the main process (section 11).
- Aggregation filled a missing std with 0. It now leaves std blank for n=1 and warns on an all-zero seed.
- The old code imported from the superproject (`src.utils`). It no longer does.
- Short test runs wrote no `last.ckpt` (save interval 10). The interval is now `min(10, epochs)`.

## 6. Deviations from the original

- Fingerprints come from a cache instead of being computed in the data loader. The values are identical (section 9).
- Batches are built in the main process (`num_workers` 0 instead of 8). The vectors sit on the GPU. Batch order and values
  are identical (section 11).
- ProtT5 pooling, tokenizer and precision follow the standard recipe. The original ships no embedding script.
- The wrapper also saves the best-val-loss checkpoint (`bestmodel.pth`). Evaluation uses the last epoch.
- Rows with a null sequence or an unfeaturisable reaction are dropped and counted.
- Protein candidates for EC ranking are the train proteins with a full EC number.
- Lightning checkpoints are slimmed to weights (no optimizer state) except `checkpoint_last.pt`.

## 7. Exact commands

Python: `/home/adhil/.conda/envs/horizyn_venv/bin/python`. It is a venv on top of mldb (`--system-site-packages`) with h5py added.
mldb has no h5py. Never install into mldb.
Run from `baselines/Horizon`. Splits found now: `with_protein_information/enzyme_sequence_splits/threshold_0.4`,
`with_protein_information/random_splits_grouped_rxn_key/all`, `with_protein_information/uniprot_time_splits/all`.
(Labels are relative to the dataset root.)

```
PY=/home/adhil/.conda/envs/horizyn_venv/bin/python
$PY -m emulator_bench.run_benchmarks --dry_run
$PY -m emulator_bench.run_benchmarks --gpus 0 1 2 --runs_per_gpu 2 --seeds 666 777 888
$PY -m emulator_bench.aggregate_results
```

One run: `--splits with_protein_information/uniprot_time_splits/all --seeds 666 --gpus 0`.
Inputs only: `--features_only`. More in `commands.txt`.

## 8. Smoke result

CPU, `uniprot_time_splits`, 64 reactions per split, deterministic protein vectors, 3 epochs, seed 666,
val and test: input build, train, evaluate, resume check and aggregation all ran. One reaction was dropped
as unfeaturisable and counted. A second smoke run used the real ProtT5 weights with two embedding shards on one A10
(`--gpus 1 1`, 64 reactions, 2 epochs): it finished and wrote all outputs.

## 9. Equivalence verdict

Test: 64 train, 16 val, 16 test reactions. Run A is the original `train.py` with `configs/sota.yaml`
(only epochs 3, validation and save interval 3, paths changed) on the same data files. Run B is the wrapper.
Both on CPU, seed 666, fp32. A Lightning callback traced each run.

| Step | Result |
|---|---|
| Featurised inputs (182 fingerprints, both directions) | identical, max abs diff 0.0 |
| Initial weights hash | identical (ca8bce9dea107557) |
| Optimizer, scheduler, loss settings | identical (AdamW 1e-4, wd 0.01, no scheduler, beta 10) |
| Batch order and tensors (3 steps) | identical ids and hashes |
| Per-step loss | identical (8.920152, 9.272566, 8.482344) |
| Epoch train loss, val loss and top-k, best epoch | identical (val/loss 6.340596, best epoch 2) |
| Final weights hash and max abs weight diff | identical, 0.0 |
| Native test metrics | identical (top_1 0, top_100 1, avg precision 0.025843) |
| EC predictions from A's checkpoint through wrapper EC code | identical to B (10 reactions) |
| Config difference | seed field in A file (42, overridden on the command line), progress bar, explicit precision 32-true (default in A) |

Rerun after the speed-ups (section 11), with `OMP_NUM_THREADS=4` for both runs: initial weights, batch hashes, per-step loss
(8.920152, 9.272566, 8.482344), weights after every step, val/loss 6.340595, final weights hash 6c5ed631e3f4e73d and native test
metrics are identical, and the max abs weight difference is 0.0. EC predictions are identical. The only config differences
are `num_workers` (8 in A, 0 in B), the seed field, the progress bar and the explicit precision. CPU results depend on the
thread count: a run with the default thread count gave other weights (max abs difference 3.7e-5), so compare with equal settings.

Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. Everything the wrapper controls matches bit for bit. The exception is the
ProtT5 embedding, which cannot be compared (no original script). The test uses a tiny set, so
large-batch behaviour (16384) is not exercised. Section 11 shows that batches of real data (the `uniprot_time_splits` validation files used as training input) match the original loader.

## 10. Open issues

- Top-level splits without a `sequence` column (random, uniprot_time, reaction DRFP threshold_0.445) are unusable. No DRFP
  or structure split with sequences exists yet. The wrapper finds them when they appear.
- The `mldb` environment has no h5py. Use `horizyn_venv` (section 7).
- ProtT5 weights `Rostlab/prot_t5_xl_half_uniref50-enc` (revision refs/pr/1) are now in the default Hugging Face cache.
  The embedding is the slowest step: about 8 sequences per second on one A10 (section 11).
- Transformers keeps the T5 `wo` layers in fp32 even for this fp16 checkpoint. The wrapper keeps that default.
- Paper lr 5e-5 and last-batch padding differ from the code. The code values are used.
- The wrapper evaluates the last epoch (original). `bestmodel.pth` is also saved.
- The screening set is all proteins of the split.
- Old results used bf16-mixed. The default now is fp32 (about 1 GB of checkpoints per run).
- Camera-ready GR-rxnsmi Horizyn error bars (L1 43.7 +- 61.7, L4 20.6 +- 29.1): the std/mean ratio is 1.41 (sqrt 2) at all
  four levels. That fits two values (x, 0) with sample std. Three seeds cannot give this (max std about 51 with mean 43.7).
  The first value matches old seed 0 (87.4, 79.1, 75.8, 41.8). So one seed was missing or scored 0, and the spread is
  an aggregation artifact, not real variance. The old per-seed files are not on disk, so this is inferred.
  A user decision is needed: rerun GR-rxnsmi with 3 seeds, or report only valid seeds.

## 11. Speed-ups

Measured on one A10 (GPU 1, shared with other jobs, so times move by 10 to 20 percent), seed 666, real split
`with_protein_information/uniprot_time_splits/all` (905,127 pairs, 77,278 train reactions, 222,243 proteins, 111 steps per epoch).
Protein vectors for the timing runs are deterministic random vectors. Training time does not depend on their values.

### Applied (exact)

| Change | Before | After | Exactness evidence |
|---|---|---|---|
| Set lookups in `TupleDataset._filter_valid_keys` (`fit.py`) | The original scans a list of 150k to 220k ids for each of 1.8M pairs. Setup did not finish in 3.4 min (estimate: more than 1 h) | data setup 50 s | Same valid keys (`test_fit_patches.py`, real split batch check) |
| Batches built in the main process, vectors on the GPU (`fit.py`, `num_workers` 0) | 8 workers: 64 to 71 s per epoch | 53.5 s per epoch (fp32) | The DataLoader still draws the same shuffle order. sha256 over 57 batches (2 epochs of train plus val loss) of the original with 8 workers and of the patch: identical (8d7dca52...). Unit test with the original `default_collate` |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the training process | 10.3 GB reserved per run | 6.4 GB reserved per run (5.84 GB peak allocated) | Allocator setting only |
| Fingerprint cache `rxnfp_v2`: three shards of float32 `[reactions, 2, 2048]` plus id lists, read with mmap | 70k small `.npy` files per split | 3 files (763 MB) | `rxn_fps.npz`, `prots_t5.h5` and all CSV files byte-identical to the old build (48,689 reactions) |
| Protein embedding in shards, one process per GPU (`run_benchmarks` with more than one GPU, `cache_features --embed_shard K/N`) | 1 process | n processes | Each protein is still embedded alone. The 93 vectors of the smoke run are bit-identical to a single process |

Epoch time stays GPU bound: the sgemm kernels use 80 percent of the GPU time. The data loader takes 13 ms per step.

### Measured and not applied (they change numbers)

| Option | Result |
|---|---|
| ProtT5 batching by length (token budgets 2048 and 8192, sorted) | No speed-up (8.8 and 8.3 sequences per second against 8.9 at batch size 1; the GPU is already full at batch size 1). Values differ: max abs 1.7e-4, max relative L2 1.2e-3, min cosine 0.9999993. Not adopted. Batch size 1 stays |
| ProtT5 with TF32 for the fp32 `wo` layers | 10.0 against 8.45 sequences per second (+19 percent). Max abs difference to the default 1.1e-4 |
| ProtT5 with `wo` in fp16 (`model.half()`) | 11.3 against 8.45 sequences per second (+34 percent). Not compared further |
| Training with TF32 (`torch.set_float32_matmul_precision("high")`) | 32.9 s against 57.9 s per epoch (1.76 times faster). Same 5.84 GB |
| Training with `--amp` (bf16 autocast) | 26.8 s per epoch (2.2 times faster), 5.44 GB |

### Full run cost (one A10, default fp32, `--runs_per_gpu 1`)

| Item | Value |
|---|---|
| Unique proteins for the three default splits | 222,477 (94.2M residues, mean length 423) |
| ProtT5 embedding | about 8.2 sequences per second per A10: about 7.5 h on one A10, about 2.5 h with three A10 shards. Done once for all splits |
| Fingerprints | 14.5 min per split with 8 workers (shared CPU), once per reaction set |
| Setup per run | about 1.5 min |
| Training epoch | 53.5 s. Validation takes 38 s and runs every 10 epochs (python loop of the original retrieval metrics, about 6.5 min per run) |
| One run, 100 epochs | about 1.6 h plus evaluation |
| GPU memory | 5.84 GB peak allocated, 6.4 GB reserved per run |

`--runs_per_gpu 2` on one A10: epoch 78 s and validation 76 s per run, so two runs finish in about 2.4 h (1.2 h per run, 1.35 times
more throughput). Three runs need 19.3 GB: they fit on an empty A10 but were not measured. A 32 GB Blackwell GPU fits four runs and
an 80 GB A100 fits more than eight. The GPU compute limits the gain, so more than two runs per GPU is unlikely to help.
The nine default runs (3 splits times 3 seeds) need about 14 to 15 GPU hours on an A10 with fp32.
