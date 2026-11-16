"""Rebuild the summary CSVs from the saved predictions and results of every run.

Writes <out_root>/enzyme_retrieval/{summary_by_seed,summary,dropped_reactions}.csv. The EC accuracy is recomputed from
predictions_<split>.csv and checked against results_<split>.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .results import compute_care_task2_metrics
from .utils import DEFAULT_OUT_ROOT, TASK, read_json


def flatten(prefix: str, data: dict) -> dict[str, float]:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(flatten(f"{prefix}.{key}", value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{prefix}.{key}"] = float(value)
    return out


def collect(task_root: Path, eval_split: str, no_b: bool = False) -> pd.DataFrame:
    rows = []
    for result_path in sorted(task_root.glob(f"**/seed_*/results_{eval_split}.json")):
        run_dir = result_path.parent
        result = read_json(result_path)
        if no_b:
            recomputed = read_json(run_dir / "ec_level_accuracy_no_b.json")
        else:
            predictions = pd.read_csv(run_dir / f"predictions_{eval_split}.csv")
            recomputed = compute_care_task2_metrics(predictions)
            if recomputed != result["ec_accuracy"]:
                raise ValueError(f"{result_path}: EC accuracy does not match the saved predictions")
        if recomputed["k=1"]["level_1_accuracy"] == 0:
            print(f"WARNING: all EC accuracies are 0 in {run_dir}. Check this run.")
        group, name = run_dir.parent.parent.relative_to(task_root).as_posix(), run_dir.parent.name
        metrics = flatten("ec", recomputed)
        if not no_b:
            metrics |= flatten("native", result["native"])
            metrics.pop("native.num_queries", None)
        rows += [
            {
                "split_group": group,
                "split_name": name,
                "seed": int(run_dir.name.removeprefix("seed_")),
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    parser.add_argument("--no_b", action="store_true", help="Summaries without preliminary ECs")
    args = parser.parse_args()
    task_root = Path(args.out_root) / TASK
    by_seed = collect(task_root, args.eval_split, args.no_b)
    suffix = "_no_b" if args.no_b else ""
    if by_seed.empty:
        raise SystemExit(f"No results found under {task_root}")
    keys = ["split_group", "split_name", "metric"]
    summary = (
        by_seed.groupby(keys)["value"].agg(n_seeds="count", mean="mean", std="std").reset_index()
    )
    by_seed.to_csv(task_root / f"summary_by_seed{suffix}.csv", index=False)
    summary.to_csv(task_root / f"summary{suffix}.csv", index=False)

    if args.no_b:
        print(f"[emulator_bench] wrote no-B summaries to {task_root}")
        return
    dropped = []
    for path in sorted(task_root.glob("**/data/data_summary.json")):
        data = read_json(path)
        dropped += [
            {
                "split": data["split"],
                "file": f,
                "reactions": s["reactions"],
                "dropped_reactions": s["dropped_unfeaturizable_reactions"],
                "dropped_pairs": s["dropped_unfeaturizable_pairs"],
                "unique_pairs": s["unique_pairs"],
            }
            for f, s in data["stats"].items()
        ]
    pd.DataFrame(dropped).to_csv(task_root / "dropped_reactions.csv", index=False)
    print(f"[emulator_bench] wrote summaries to {task_root}")


if __name__ == "__main__":
    main()
