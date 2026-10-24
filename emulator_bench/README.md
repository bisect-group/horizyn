# Horizon (Horizyn) EMULaToR wrapper

This wrapper trains Horizyn with its original settings (`configs/sota.yaml`, 100 epochs) on the
EMULaToR enzyme retrieval splits. It runs seeds 666, 777 and 888 on every split. It does no tuning.
Defaults and their sources are in `default_hparams.json`. See `REPORT.md` for the audit.

## Data

Horizyn needs reaction-protein pairs. Only split folders whose parquet files hold a `sequence`
column work. Today this means the folders under `with_protein_information/`. The wrapper finds the
folders on disk (`train.parquet`, `val.parquet`, `test.parquet`). It never re-splits.

- `rxn_smiles` is the reaction. `sequence` is the protein. Repeated pairs become one pair.
- All pairs train and validate, also those with unknown or partial EC numbers.
- `ec_number` is used only in evaluation. Only four-level EC numbers count.
- Sequences are not cut. ProtT5 gets chunks of 5000 residues. The chunk vectors are averaged.
- Reactions that the Horizyn featurizer cannot process are dropped and counted
  (`dropped_reactions.csv`). Dropped test reactions with a full EC count as misses in the EC accuracy
  (`reactions_missed_unfeaturizable` in `results_test.json`). The native metrics skip them. Rows without a sequence are counted as invalid rows.

## Output

Everything goes to `<EMULaToR>/outputs/baselines/Horizon` (`--out_root`). The dataset tree is read only.

- `embeddings/prott5_v1/` (one `.npy` per protein), `embeddings/rxnfp_v2/` (three shards of reaction fingerprints, read with mmap): shared caches
- `enzyme_retrieval/<split_group>/<split_name>/data/`: pair files, protein vectors, fingerprints
- `enzyme_retrieval/<split_group>/<split_name>/seed_<seed>/`: `bestmodel.pth`, `lastmodel.pth`,
  `checkpoint_last.pt`, `config.yaml`, `logs/`, `predictions_test.csv`, `results_test.json`
- `enzyme_retrieval/{summary_by_seed,summary,dropped_reactions}.csv`

## Run

Run from this repository root (`baselines/Horizon`). See `emulator_bench/commands.txt`.

```
python -m emulator_bench.run_benchmarks --gpus 0 1 2 --runs_per_gpu 2 [--amp]
python -m emulator_bench.aggregate_results
```

- `--amp` uses bf16 autocast (fp16 on older GPUs). Without it the run uses fp32, as the original.
- A finished run has `results_test.json` and is skipped. An interrupted run resumes from `checkpoint_last.pt`.
  Lightning does not restore the global RNG on resume, so a resumed run is valid but not bit-equal to an uninterrupted one.
  A run that stops before the last epoch (for example Ctrl+C) fails and is not exported as finished.
- Without `--gpus`, runs use the CPU (`CUDA_VISIBLE_DEVICES` is empty for every child process).
- The environment needs `torch`, `lightning`, `rdkit`, `drfp`, `h5py`, `transformers`, `sentencepiece`,
  `pandas`, `pyarrow`. Use `/home/adhil/.conda/envs/horizyn_venv/bin/python` (a venv on top of mldb with h5py).
- ProtT5 weights: `Rostlab/prot_t5_xl_half_uniref50-enc` (about 2.4 GB) must be in the Hugging Face cache.
- With more than one GPU, each GPU embeds one shard of the proteins. Each protein is still embedded alone.
- Batches are built in the main process with the vectors on the GPU (`num_workers` 0). A run needs about 6.5 GB of GPU memory.
  Two runs per GPU give about 1.35 times the throughput of one.

## Evaluation

The run is evaluated at the last epoch, as in the original README. Each reaction ranks the EC numbers
of all proteins of the split (train, val and test, also those of dropped reactions; `data/pool_manifest.csv`) by their best protein score. Forward and reverse scores combine with max.
The paper number is the EC accuracy at k=1 per level (L1 to L4). The native Horizyn metrics
(top-k, R-precision, average precision) are saved in `results_test.json`.
