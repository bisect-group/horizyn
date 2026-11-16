"""Write ec_level_accuracy_no_b.json for every finished run from its saved predictions_<split>.csv.

Default: next to the predictions. With --dest_root the files go to a mirror of the run folders instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .results import compute_ec_accuracy_no_b
from .utils import DEFAULT_OUT_ROOT, TASK, atomic_write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--dest_root", default=None)
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    task_root = Path(args.out_root) / TASK
    for path in sorted(task_root.glob(f"**/seed_*/predictions_{args.eval_split}.csv")):
        dest = path.parent
        if args.dest_root:
            dest = Path(args.dest_root) / TASK / path.parent.relative_to(task_root)
        metrics = compute_ec_accuracy_no_b(pd.read_csv(path))
        atomic_write(
            dest / "ec_level_accuracy_no_b.json",
            lambda tmp: Path(tmp).write_text(json.dumps(metrics, indent=2)),
        )
        print(f"[emulator_bench] {path.parent.relative_to(task_root)}: {metrics['n_queries']} queries")


if __name__ == "__main__":
    main()
