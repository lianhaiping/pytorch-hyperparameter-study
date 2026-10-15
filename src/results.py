"""Result files for one stage:

results/<stage>/runs.jsonl                 one summary line per finished run
results/<stage>/histories/<run_id>.csv     per-epoch train/val RMSE, steps, time
results/<stage>/val_preds/<run_id>.npy     best-model validation predictions (float32)
results/<stage>/env.json                   software/hardware environment
"""

import csv
import json
from pathlib import Path

import numpy as np

HISTORY_FIELDS = ["epoch", "steps", "samples_seen", "train_time_s", "train_rmse", "val_rmse"]


def read_runs(stage_dir: Path) -> list[dict]:
    path = Path(stage_dir) / "runs.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_run(stage_dir: Path, summary: dict, history: list[dict], val_pred: np.ndarray) -> None:
    stage_dir = Path(stage_dir)
    (stage_dir / "histories").mkdir(parents=True, exist_ok=True)
    (stage_dir / "val_preds").mkdir(parents=True, exist_ok=True)
    run_id = summary["run_id"]

    # History and predictions are written before the summary line, so a run that
    # appears in runs.jsonl always has its companion files (safe to resume).
    with open(stage_dir / "histories" / f"{run_id}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    np.save(stage_dir / "val_preds" / f"{run_id}.npy", val_pred)
    with open(stage_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
        f.flush()


def read_history(stage_dir: Path, run_id: str) -> list[dict]:
    with open(Path(stage_dir) / "histories" / f"{run_id}.csv", newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_env(stage_dir: Path, env: dict) -> None:
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(stage_dir) / "env.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
