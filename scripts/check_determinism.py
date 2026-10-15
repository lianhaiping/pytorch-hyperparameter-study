"""Run one configuration twice and verify the results are bit-identical.

    python scripts/check_determinism.py --optimizer adam --batch-size 256 --lr 1e-3 --epochs 20

Exits with status 0 on PASS, 1 on FAIL. Writes nothing to results/.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_train_val  # noqa: E402
from src.repro import resolve_device  # noqa: E402
from src.train import RunSpec, train_one  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "sweep.toml"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--optimizer", default="adam", choices=("sgd", "adam"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20, help="max_epochs override to keep the check short")
    args = p.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    cfg["training"]["max_epochs"] = args.epochs
    device = resolve_device(args.device)
    data = load_train_val(cfg, ROOT)
    spec = RunSpec(args.optimizer, args.batch_size, args.lr, cfg["training"]["seed"])

    s1, h1, p1 = train_one(spec, data, cfg, device, stage="check")
    s2, h2, p2 = train_one(spec, data, cfg, device, stage="check")

    timing_keys = {"train_time_to_best_s", "total_train_time_s"}
    failures = []
    if {k: v for k, v in s1.items() if k not in timing_keys} != {k: v for k, v in s2.items() if k not in timing_keys}:
        failures.append("summaries differ")
    for a, b in zip(h1, h2, strict=True):
        # equal_nan: a diverged epoch is recorded as NaN in both runs.
        if not np.array_equal([a["train_rmse"], a["val_rmse"], a["steps"]],
                              [b["train_rmse"], b["val_rmse"], b["steps"]], equal_nan=True):
            failures.append(f"history differs at epoch {a['epoch']}")
            break
    if not np.array_equal(p1, p2):
        failures.append("validation predictions differ")

    print(f"{spec.run_id} on {device}, {len(h1) - 1} epochs, best_val_rmse={s1['best_val_rmse']:.6f}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: two runs are bit-identical (timing excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
