"""Process-level helpers so runs can execute in parallel worker processes.

Each worker loads the data once and uses one CPU thread per run
([training].num_threads), so parallel runs stay deterministic.
"""

from pathlib import Path

from .data import load_train_val
from .repro import resolve_device
from .train import RunSpec, train_one

_WORKER: dict = {}


def init_worker(cfg: dict, root: str, device_name: str) -> None:
    _WORKER["cfg"] = cfg
    _WORKER["data"] = load_train_val(cfg, Path(root))
    _WORKER["device"] = resolve_device(device_name)


def run_spec(spec: RunSpec, stage: str):
    return train_one(spec, _WORKER["data"], _WORKER["cfg"], _WORKER["device"], stage)
