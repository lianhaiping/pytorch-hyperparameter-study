"""Pre-registered rule for choosing the best learning rate per (optimizer, batch size)."""

import math
from collections import defaultdict


def is_eligible(run: dict) -> bool:
    # best_epoch == 0 means the model never improved on its untrained weights.
    return run["best_epoch"] >= 1 and math.isfinite(run["best_val_rmse"])


def group_runs(runs: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["optimizer"], r["batch_size"])].append(r)
    return dict(groups)


def select_best(runs: list[dict], tie_rel_tol: float) -> dict[tuple[str, int], dict]:
    """For each (optimizer, batch size): the eligible run with the lowest validation
    RMSE. Runs within tie_rel_tol (relative) of that minimum count as ties, and the
    largest LR among them wins. Pass runs from a single seed.

    Each returned record is a copy with an extra key 'lr_at_edge': 'low' or 'high'
    if the chosen LR is the smallest/largest LR tested for that group, else ''.
    """
    seeds = {r["seed"] for r in runs}
    if len(seeds) > 1:
        raise ValueError(f"select_best expects a single seed, got {sorted(seeds)}")

    best = {}
    for key, group in group_runs(runs).items():
        eligible = [r for r in group if is_eligible(r)]
        if not eligible:
            continue
        min_rmse = min(r["best_val_rmse"] for r in eligible)
        ties = [r for r in eligible if r["best_val_rmse"] <= min_rmse * (1.0 + tie_rel_tol)]
        chosen = dict(max(ties, key=lambda r: r["lr"]))
        tested_lrs = [r["lr"] for r in group]
        if math.isclose(chosen["lr"], min(tested_lrs)):
            chosen["lr_at_edge"] = "low"
        elif math.isclose(chosen["lr"], max(tested_lrs)):
            chosen["lr_at_edge"] = "high"
        else:
            chosen["lr_at_edge"] = ""
        best[key] = chosen
    return best


def dedupe(runs: list[dict]) -> list[dict]:
    """A run is fully determined by (optimizer, batch size, lr, seed), so the same run_id
    finished in two stages is the same result; keep the first occurrence."""
    seen, out = set(), []
    for r in runs:
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            out.append(r)
    return out


def mean_over_seeds(runs: list[dict], seeds: list[int]) -> list[dict]:
    """One record per (optimizer, batch size, lr) that has a run for EVERY seed in `seeds`,
    with best_val_rmse = mean over those seeds. best_epoch is the minimum over seeds, so a
    configuration counts as eligible only if it improved on its initial weights for every
    seed. Incomplete configurations are skipped. The result has seed='mean' and can be
    passed to select_best."""
    wanted = set(seeds)
    by_cfg: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in runs:
        if r["seed"] in wanted:
            by_cfg[(r["optimizer"], r["batch_size"], r["lr"])][r["seed"]] = r
    out = []
    for (opt, bs, lr), per_seed in by_cfg.items():
        if set(per_seed) != wanted:
            continue
        vals = [per_seed[s]["best_val_rmse"] for s in seeds]
        out.append(dict(run_id=f"{opt}_bs{bs}_lr{lr:.4g}_mean", optimizer=opt, batch_size=bs, lr=lr,
                        seed="mean", n_seeds=len(seeds), best_val_rmse=sum(vals) / len(vals),
                        best_epoch=min(per_seed[s]["best_epoch"] for s in seeds)))
    return out


def on_half_decade_grid(lr: float) -> bool:
    """True for 1-3-10 grid values (1e-4, 3e-4, 1e-3, ...); False for refine points."""
    exponent = math.floor(math.log10(lr) + 1e-9)
    return math.isclose(lr / 10 ** exponent, 1.0) or math.isclose(lr / 10 ** exponent, 3.0)


def next_half_decade(lr: float, direction: str) -> float:
    """Neighbour on the 1-3-10 grid (..., 1e-3, 3e-3, 1e-2, ...)."""
    exponent = math.floor(math.log10(lr) + 1e-9)
    mantissa = round(lr / 10 ** exponent)
    if mantissa not in (1, 3):
        raise ValueError(f"{lr} is not on the 1-3-10 half-decade grid")
    if direction == "up":
        value = 3 * 10 ** exponent if mantissa == 1 else 10 ** (exponent + 1)
    elif direction == "down":
        value = 10 ** exponent if mantissa == 3 else 3 * 10 ** (exponent - 1)
    else:
        raise ValueError("direction must be 'up' or 'down'")
    return float(f"{value:.0e}")  # strip float noise, e.g. 2.9999999999999997e-06 -> 3e-06
