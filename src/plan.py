"""Which runs each stage performs. Seed is a full factor: every stage runs every seed.

main      full grid: optimizer x batch size x every LR in lr_grid x every seed
extend    pre-registered interior-optimum rule: where a group's best LR (by the mean over
          seeds) sits on the edge of the tested half-decade grid, add the next half-decade LR
          beyond it, for every seed. Re-run this stage until it plans 0 runs.
refine    best LR (by the mean over seeds) x each factor in [refine].factors, for every seed
pilot     timing pilot on the reference seed (results are not used in the analysis)
capcheck  sensitivity check: runs that hit the epoch cap at batch sizes >= [capcheck].min_batch_size,
          re-run with [capcheck].max_epochs (reported separately; not part of the main results)

Selection for extend/refine uses the mean over ALL seeds. It only decides which extra LRs
to run; the primary results use leave-one-seed-out selection over the half-decade grid
(see analysis.py). Refine points are used only in supporting analyses (e.g. LR scaling).
"""

import math

from .selection import (group_runs, is_eligible, mean_over_seeds, next_half_decade,
                        on_half_decade_grid, select_best)
from .train import RunSpec

STAGES = ("pilot", "main", "extend", "refine", "capcheck")


def plan(stage: str, cfg: dict, prior_runs: list[dict]) -> list[RunSpec]:
    """prior_runs: deduplicated runs from every stage except pilot."""
    seeds = cfg["sweep"]["seeds"]
    if stage == "pilot":
        seed = cfg["training"]["seed"]
        return [RunSpec(opt, bs, cfg["pilot"]["lr"][opt], seed)
                for opt in cfg["optimizers"] for bs in cfg["pilot"]["batch_sizes"]]
    if stage == "main":
        return [RunSpec(opt, bs, lr, s)
                for s in seeds
                for opt, oc in cfg["optimizers"].items()
                for bs in cfg["sweep"]["batch_sizes"]
                for lr in oc["lr_grid"]]
    if stage not in STAGES:
        raise ValueError(f"unknown stage '{stage}'; expected one of {STAGES}")

    _require_complete_main(cfg, prior_runs)
    if stage == "capcheck":
        cc = cfg["capcheck"]
        return [RunSpec(r["optimizer"], r["batch_size"], r["lr"], r["seed"]) for r in prior_runs
                if r["hit_cap"] and r["batch_size"] >= cc["min_batch_size"]]
    tie_rel_tol = cfg["selection"]["tie_rel_tol"]
    agg = mean_over_seeds(prior_runs, seeds)
    if stage == "extend":
        return _plan_extend(agg, seeds, tie_rel_tol)

    # refine: extra LRs around each group's best half-decade LR (mean over seeds).
    best = select_best([a for a in agg if on_half_decade_grid(a["lr"])], tie_rel_tol)
    specs = []
    for (opt, bs), rec in sorted(best.items()):
        tested = [a["lr"] for a in agg if a["optimizer"] == opt and a["batch_size"] == bs]
        for factor in cfg["refine"]["factors"]:
            lr = float(f"{rec['lr'] * factor:.6g}")
            if not any(math.isclose(lr, t) for t in tested):
                specs += [RunSpec(opt, bs, lr, s) for s in seeds]
    return specs


def _require_complete_main(cfg: dict, prior_runs: list[dict]) -> None:
    done = {r["run_id"] for r in prior_runs}
    missing = [s.run_id for s in plan("main", cfg, []) if s.run_id not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} main-grid runs are not finished yet "
                           f"(e.g. {missing[0]}); run --stage main first")


def _plan_extend(agg: list[dict], seeds: list[int], tie_rel_tol: float) -> list[RunSpec]:
    # Only half-decade LRs define the grid edges; refine points (x0.5, x2) do not.
    grid = [a for a in agg if on_half_decade_grid(a["lr"])]
    specs = []
    for (opt, bs), group in sorted(group_runs(grid).items()):
        tested = sorted(a["lr"] for a in group)
        if not any(is_eligible(a) for a in group):
            # Every LR diverged or never improved on some seed: probe below the smallest LR.
            new_lr = next_half_decade(tested[0], "down")
        else:
            rec = select_best(group, tie_rel_tol)[(opt, bs)]
            if rec["lr_at_edge"] == "low":
                new_lr = next_half_decade(rec["lr"], "down")
            elif rec["lr_at_edge"] == "high":
                new_lr = next_half_decade(rec["lr"], "up")
            else:
                continue
        if not any(math.isclose(new_lr, t) for t in tested):
            specs += [RunSpec(opt, bs, new_lr, s) for s in seeds]
    return specs
