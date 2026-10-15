"""Load and validate the TOML study configuration."""

import hashlib
import json
import tomllib
from pathlib import Path

OPTIMIZERS = ("sgd", "adam")


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    _validate(cfg)
    return cfg


def settings_fingerprint(cfg: dict, optimizer: str) -> str:
    """Short hash of every setting that can change a run's result but is NOT part of its
    run_id (optimizer, batch size, LR, seed): the data split, the model, the training settings,
    and that optimizer's settings. Runs with different fingerprints must not be mixed.

    Excluded on purpose: training.seed (only the default for the pilot and checks; each run's
    seed is in its run_id) and lr_grid (decides which runs exist, not how a run trains)."""
    training = {k: v for k, v in cfg["training"].items() if k != "seed"}
    opt = {k: v for k, v in cfg["optimizers"][optimizer].items() if k != "lr_grid"}
    payload = {"data": cfg["data"], "model": cfg["model"], "training": training, "optimizer": opt}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _validate(cfg: dict) -> None:
    for section in ("data", "model", "training", "optimizers", "sweep", "selection"):
        if section not in cfg:
            raise ValueError(f"config is missing section [{section}]")

    for name in cfg["optimizers"]:
        if name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer '{name}'; expected one of {OPTIMIZERS}")
        grid = cfg["optimizers"][name]["lr_grid"]
        if sorted(grid) != grid or len(set(grid)) != len(grid) or min(grid) <= 0:
            raise ValueError(f"lr_grid for '{name}' must be positive, strictly increasing")
        # The study disables explicit regularization; refuse configs that re-enable it silently.
        if cfg["optimizers"][name].get("weight_decay", 0.0) != 0.0:
            raise ValueError(f"weight_decay for '{name}' must be 0.0 in this study")

    batch_sizes = cfg["sweep"]["batch_sizes"]
    if sorted(batch_sizes) != batch_sizes or min(batch_sizes) < 1:
        raise ValueError("batch_sizes must be positive and strictly increasing")
