import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(mo.notebook_dir() or Path.cwd())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.config import load_config
    from src.plan import plan
    from src.results import read_history, read_runs
    from src.selection import dedupe, mean_over_seeds, on_half_decade_grid, select_best
    return (ROOT, dedupe, load_config, math, mean_over_seeds, mo, np, on_half_decade_grid, pd, plan, plt,
            read_history, read_runs, select_best)


@app.cell
def _(mo):
    mo.md(r"""
    # Batch size study: California Housing, PyTorch MLP

    Does *"Friends dont let friends use minibatches larger than 32"* (LeCun, 2018) hold for
    a small tabular regression MLP when the learning rate is tuned fairly for every batch size?

    **Factors:** batch size (powers of 2), learning rate, optimizer, and seed (weight
    initialization + shuffle order). Every configuration is run with every seed.

    **All metrics are validation RMSE** (target in units of $100k). The test split is never
    used. Explicit regularization is off and the learning rate is constant.

    **Protocol (leave-one-seed-out):** for each seed in turn, the best LR for every
    (optimizer, batch size) is chosen on the *mean over the other seeds*, then scored on the
    held-out seed. Each batch size therefore gets one unbiased score per seed, and no score
    comes from the data used to choose its LR. Selection uses the half-decade LR grid
    (plus any `extend` points).

    This notebook only reads files in `results/`; it never trains.
    """)
    return


@app.cell
def _(ROOT, dedupe, load_config, mo, on_half_decade_grid, pd, plan, read_runs):
    CFG = load_config(ROOT / "configs" / "sweep.toml")
    RESULTS = ROOT / "results"
    TIE = CFG["selection"]["tie_rel_tol"]

    _all = dedupe([r for _s in ("main", "extend", "refine", "seedcheck") for r in read_runs(RESULTS / _s)])
    mo.stop(not _all, mo.md("**No results yet.** Run `python run_sweep.py --stage main` first."))
    runs = pd.DataFrame(_all)

    # A seed counts as complete once its whole main grid is finished.
    _done = set(runs["run_id"])
    _main = plan("main", CFG, [])
    SEEDS = [s for s in CFG["sweep"]["seeds"]
             if all(sp.run_id in _done for sp in _main if sp.seed == s)]
    grid_runs = [r for r in _all if on_half_decade_grid(r["lr"]) and r["seed"] in SEEDS]
    seed_runs_all = [r for r in _all if r["seed"] in SEEDS]  # grid + refine points (LR scaling only)
    mo.md(f"Loaded {len(runs)} runs. Seeds with a complete main grid: **{SEEDS}** "
          f"(configured: {CFG['sweep']['seeds']}).")
    return CFG, RESULTS, SEEDS, TIE, grid_runs, runs, seed_runs_all


@app.cell
def _(SEEDS, mo, runs):
    _r = runs[runs["seed"].isin(SEEDS)]
    _status = _r.groupby(["optimizer", "seed", "stop_reason"]).size().unstack(fill_value=0).reset_index()
    mo.vstack([
        mo.md("## Run status\n`max_epochs` = hit the epoch cap (not converged), `diverged` = stopped "
              "by the divergence rule, `patience` = converged (early stopping)."),
        mo.ui.table(_status, selection=None),
    ])
    return


@app.cell
def _(SEEDS, TIE, grid_runs, mean_over_seeds, mo, pd, select_best):
    mo.stop(len(SEEDS) < 2, mo.md("## Primary result\nLeave-one-seed-out needs at least 2 complete seeds."))
    _lookup = {(r["optimizer"], r["batch_size"], r["lr"], r["seed"]): r for r in grid_runs}
    _rows = []
    for _h in SEEDS:
        _others = [s for s in SEEDS if s != _h]
        _chosen = select_best(mean_over_seeds(grid_runs, _others), TIE)
        for (_opt, _bs), _rec in _chosen.items():
            _run = _lookup[(_opt, _bs, _rec["lr"], _h)]
            _rows.append(dict(optimizer=_opt, batch_size=_bs, heldout_seed=_h, lr=_rec["lr"],
                              lr_at_edge=_rec["lr_at_edge"], rmse=_run["best_val_rmse"],
                              stop_reason=_run["stop_reason"]))
    loso = pd.DataFrame(_rows).sort_values(["optimizer", "batch_size", "heldout_seed"]).reset_index(drop=True)
    loso_agg = (loso.groupby(["optimizer", "batch_size"])
                .agg(mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"), n_seeds=("rmse", "count"),
                     lrs_chosen=("lr", lambda x: ", ".join(f"{v:g}" for v in sorted(set(x)))),
                     n_distinct_lrs=("lr", "nunique"))
                .reset_index())
    return loso, loso_agg


@app.cell
def _(loso, loso_agg, mo, plt):
    _fig, _ax = plt.subplots(figsize=(8.5, 5))
    for _opt, _g in loso_agg.groupby("optimizer"):
        _line = _ax.errorbar(_g["batch_size"], _g["mean_rmse"], yerr=_g["std_rmse"], marker="o",
                             capsize=3, label=f"{_opt}: mean ± std over held-out seeds")
        _pts = loso[loso["optimizer"] == _opt]
        _ax.scatter(_pts["batch_size"], _pts["rmse"], s=10, alpha=0.35, color=_line[0].get_color())
    _ax.axvline(32, ls="--", color="gray", label="batch size 32")
    _ticks = sorted(loso_agg["batch_size"].unique())
    _ax.set_xscale("log", base=2)
    _ax.set_xticks(_ticks, [str(t) for t in _ticks])
    _ax.set_xlabel("batch size")
    _ax.set_ylabel("validation RMSE at selected LR ($100k)")
    _ax.set_title("Tuned RMSE vs batch size (leave-one-seed-out; dots = individual seeds)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("## Primary result: tuned RMSE vs batch size"), _fig,
               mo.ui.table(loso_agg, selection=None)])
    return


@app.cell
def _(loso, mo, pd, plt):
    # Every batch size vs 32, paired by held-out seed. diff = RMSE(bs) - RMSE(32):
    # negative means that batch size beats 32 on that seed.
    _rows = []
    for _opt, _g in loso.groupby("optimizer"):
        _piv = _g.pivot_table(index="heldout_seed", columns="batch_size", values="rmse")
        if 32 not in _piv:
            continue
        for _bs in _piv.columns:
            _d = (_piv[_bs] - _piv[32]).dropna()
            _rows.append(dict(optimizer=_opt, batch_size=int(_bs), mean_rmse=_piv[_bs].mean(),
                              mean_diff_vs_32=_d.mean(), std_diff=_d.std(),
                              diff_pct=100 * _d.mean() / _piv[32].mean(),
                              beats_32="reference" if _bs == 32 else f"{int((_d < 0).sum())} of {len(_d)} seeds"))
    vs32 = pd.DataFrame(_rows)

    _fig, _ax = plt.subplots(figsize=(8.5, 4.5))
    for _opt, _g in vs32.groupby("optimizer"):
        _ax.errorbar(_g["batch_size"], _g["mean_diff_vs_32"], yerr=_g["std_diff"], marker="o",
                     capsize=3, label=_opt)
    _ax.axhline(0, color="gray", lw=1)
    _ax.axvline(32, ls="--", color="gray", label="batch size 32")
    _ticks = sorted(vs32["batch_size"].unique())
    _ax.set_xscale("log", base=2)
    _ax.set_xticks(_ticks, [str(t) for t in _ticks])
    _ax.set_xlabel("batch size")
    _ax.set_ylabel("RMSE(bs) - RMSE(32), mean ± std over seeds")
    _ax.set_title("Each batch size vs 32, paired by held-out seed (below 0 = better than 32)")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md("## Each batch size vs 32 (paired by seed)"), _fig,
               mo.ui.table(vs32, selection=None)])
    return (vs32,)


@app.cell
def _(SEEDS, best_mean, grid_runs, mo, pd, plt):
    # Cost of NOT retuning: keep the LR that is best for batch size 32 (seed mean) and use it
    # at every batch size, vs each batch size's own best LR. Mean RMSE over seeds.
    mo.stop(not SEEDS or 32 not in set(best_mean["batch_size"]))
    _cell = (pd.DataFrame(grid_runs).groupby(["optimizer", "batch_size", "lr"])["best_val_rmse"]
             .mean().rename("rmse").reset_index())
    _rows = []
    for _opt, _g in best_mean.groupby("optimizer"):
        _lr32 = _g.loc[_g["batch_size"] == 32, "lr"].iloc[0]
        for _rec in _g.to_dict("records"):
            _fixed = _cell[(_cell["optimizer"] == _opt) & (_cell["batch_size"] == _rec["batch_size"])
                           & (_cell["lr"] == _lr32)]["rmse"].iloc[0]
            _rows.append(dict(optimizer=_opt, batch_size=_rec["batch_size"], lr_bs32=_lr32,
                              rmse_tuned_lr=_rec["best_val_rmse"], rmse_with_lr_of_bs32=_fixed,
                              cost_pct=100 * (_fixed / _rec["best_val_rmse"] - 1)))
    no_retune = pd.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(8.5, 4.5))
    for _opt, _g in no_retune.groupby("optimizer"):
        _ax.plot(_g["batch_size"], _g["cost_pct"], "o-", label=_opt)
    _ax.axvline(32, ls="--", color="gray", label="batch size 32")
    _ticks = sorted(no_retune["batch_size"].unique())
    _ax.set_xscale("log", base=2)
    _ax.set_yscale("symlog", linthresh=5)
    _ax.set_xticks(_ticks, [str(t) for t in _ticks])
    _ax.set_xlabel("batch size")
    _ax.set_ylabel("% worse RMSE than with a retuned LR")
    _ax.set_title("Cost of keeping the batch-size-32 learning rate (mean over seeds)")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md("## Cost of not retuning the learning rate when batch size changes"), _fig,
               mo.ui.table(no_retune, selection=None)])
    return (no_retune,)


@app.cell
def _(SEEDS, TIE, grid_runs, mo, pd, plt, select_best):
    mo.stop(len(SEEDS) < 2)
    # How stable is the best LR across seeds? Each seed's own best LR (selected on that seed).
    _rows = []
    for _s in SEEDS:
        for (_opt, _bs), _rec in select_best([r for r in grid_runs if r["seed"] == _s], TIE).items():
            _rows.append(dict(optimizer=_opt, batch_size=_bs, seed=_s, lr=_rec["lr"]))
    _per_seed = pd.DataFrame(_rows)
    _opts = sorted(_per_seed["optimizer"].unique())
    _fig, _axes = plt.subplots(1, len(_opts), figsize=(6.5 * len(_opts), 4.5), squeeze=False)
    # Small horizontal offsets so overlapping seeds stay visible.
    _jit = {s: 2 ** (0.06 * (i - (len(SEEDS) - 1) / 2)) for i, s in enumerate(SEEDS)}
    for _ax, _opt in zip(_axes[0], _opts):
        _g = _per_seed[_per_seed["optimizer"] == _opt]
        for _s, _gs in _g.groupby("seed"):
            _ax.plot(_gs["batch_size"] * _jit[_s], _gs["lr"], "o", alpha=0.7, label=f"seed {_s}")
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log")
        _ax.set_xlabel("batch size")
        _ax.set_ylabel("best LR on that seed")
        _ax.set_title(f"{_opt}: best LR per seed")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _table = (_per_seed.pivot_table(index=["optimizer", "batch_size"], columns="seed", values="lr")
              .reset_index())
    _table.columns = [str(c) for c in _table.columns]
    mo.vstack([mo.md("## Stability of the best learning rate across seeds"), _fig,
               mo.ui.table(_table, selection=None)])
    return


@app.cell
def _(SEEDS, TIE, grid_runs, mean_over_seeds, mo, pd, select_best):
    mo.stop(not SEEDS, mo.md("Supporting analyses need at least one complete seed."))
    # Supporting analyses use the LR that is best on the mean over ALL complete seeds.
    agg_all = pd.DataFrame(mean_over_seeds(grid_runs, SEEDS))
    best_mean = (pd.DataFrame(list(select_best(agg_all.to_dict("records"), TIE).values()))
                 .sort_values(["optimizer", "batch_size"]).reset_index(drop=True))
    return agg_all, best_mean


@app.cell
def _(SEEDS, best_mean, grid_runs, mo, np, pd, plt):
    # Mean and std over seeds for every (batch size, LR) cell of the grid.
    _df = pd.DataFrame(grid_runs)
    _cell = _df.groupby(["optimizer", "batch_size", "lr"])["best_val_rmse"].agg(["mean", "std"]).reset_index()
    _opts = sorted(_cell["optimizer"].unique())
    _fig, _axes = plt.subplots(2, len(_opts), figsize=(6.5 * len(_opts), 9), squeeze=False)
    for _col, _opt in enumerate(_opts):
        for _row, (_stat, _title) in enumerate([("mean", "mean RMSE"), ("std", "std of RMSE across seeds")]):
            _ax = _axes[_row][_col]
            _piv = _cell[_cell["optimizer"] == _opt].pivot_table(index="batch_size", columns="lr", values=_stat)
            _vals = _piv.to_numpy(dtype=float)
            _im = _ax.imshow(np.clip(_vals, None, np.nanpercentile(_vals, 90)), aspect="auto",
                             cmap="viridis_r" if _stat == "mean" else "magma_r", origin="lower")
            _ax.set_xticks(range(len(_piv.columns)), [f"{c:.0e}" for c in _piv.columns], rotation=45)
            _ax.set_yticks(range(len(_piv.index)), _piv.index)
            _ax.set_xlabel("learning rate")
            _ax.set_ylabel("batch size")
            _ax.set_title(f"{_opt}: {_title} (colour clipped at 90th pct)")
            if _stat == "mean":
                for _i, _bs in enumerate(_piv.index):
                    _b = best_mean[(best_mean["optimizer"] == _opt) & (best_mean["batch_size"] == _bs)]
                    if len(_b) and _b["lr"].iloc[0] in _piv.columns:
                        _ax.plot(list(_piv.columns).index(_b["lr"].iloc[0]), _i, "*", color="red", markersize=12)
            _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    mo.vstack([mo.md(f"## Batch size x learning rate over {len(SEEDS)} seeds "
                     "(red star = best LR on the mean over seeds)"), _fig])
    return


@app.cell
def _(CFG, SEEDS, TIE, best_mean, math, mean_over_seeds, mo, np, pd, plt, seed_runs_all, select_best):
    # LR scaling: slope of log10(best LR) vs log10(batch size).
    # ~1 matches the linear scaling rule (SGD, Goyal et al. 2017), ~0.5 the square-root rule
    # (adaptive methods, Malladi et al. 2022).
    # Candidates per (optimizer, batch size): the half-decade grid plus the refine points
    # (x0.5, x2 around the seed-mean best grid LR), i.e. roughly doubled LR resolution.
    # Slopes are fitted over all batch sizes and over batch sizes <= 512 (before SGD's LR ceiling),
    # both on the seed-mean best LR and separately per seed (mean ± std across seeds).
    def _slope(g, max_bs=None):
        g = g if max_bs is None else g[g["batch_size"] <= max_bs]
        return np.polyfit(np.log10(g["batch_size"]), np.log10(g["lr"]), 1)[0]

    _allowed = {}
    for _rec in best_mean.to_dict("records"):
        _allowed[(_rec["optimizer"], _rec["batch_size"])] = [_rec["lr"] * f for f in CFG["refine"]["factors"]]

    def _is_candidate(r):
        from src.selection import on_half_decade_grid
        return on_half_decade_grid(r["lr"]) or any(
            math.isclose(r["lr"], a, rel_tol=1e-6) for a in _allowed.get((r["optimizer"], r["batch_size"]), []))

    _cand = [r for r in seed_runs_all if _is_candidate(r)]
    best_refined = (pd.DataFrame(list(select_best(mean_over_seeds(_cand, SEEDS), TIE).values()))
                    .sort_values(["optimizer", "batch_size"]).reset_index(drop=True))
    _per_seed = pd.DataFrame([dict(optimizer=o, batch_size=b, seed=s, lr=rec["lr"])
                              for s in SEEDS
                              for (o, b), rec in select_best([r for r in _cand if r["seed"] == s], TIE).items()])
    _rows = []
    for _opt, _g in best_refined.groupby("optimizer"):
        _ps = _per_seed[_per_seed["optimizer"] == _opt]
        _all = [_slope(_gs) for _, _gs in _ps.groupby("seed")]
        _low = [_slope(_gs, 512) for _, _gs in _ps.groupby("seed")]
        _rows.append(dict(optimizer=_opt,
                          slope_seed_mean_all=_slope(_g), slope_seed_mean_le512=_slope(_g, 512),
                          per_seed_all=f"{np.mean(_all):.2f} ± {np.std(_all, ddof=1) if len(_all) > 1 else 0:.2f}",
                          per_seed_le512=f"{np.mean(_low):.2f} ± {np.std(_low, ddof=1) if len(_low) > 1 else 0:.2f}",
                          n_at_edge=int((_g["lr_at_edge"] != "").sum())))
    lr_slopes = pd.DataFrame(_rows)

    _fig, _ax = plt.subplots(figsize=(8.5, 5))
    for _opt, _g in best_refined.groupby("optimizer"):
        _ax.plot(_g["batch_size"], _g["lr"], "o-", label=f"{_opt} (slope {_slope(_g):.2f})")
    _x = np.array([best_refined["batch_size"].min(), best_refined["batch_size"].max()], dtype=float)
    _y0 = best_refined["lr"].min()
    _ax.plot(_x, _y0 * _x / _x[0], ":", color="gray", label="slope 1 (linear rule), reference")
    _ax.plot(_x, _y0 * np.sqrt(_x / _x[0]), "--", color="gray", alpha=0.6, label="slope 0.5 (square-root rule)")
    _ticks = sorted(best_refined["batch_size"].unique())
    _ax.set_xscale("log", base=2)
    _ax.set_yscale("log")
    _ax.set_xticks(_ticks, [str(t) for t in _ticks])
    _ax.set_ylim(_y0 / 3, best_refined["lr"].max() * 3)
    _ax.set_xlabel("batch size")
    _ax.set_ylabel("best learning rate (mean over seeds; grid + refine)")
    _ax.set_title("Best learning rate vs batch size")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("## Learning-rate scaling with batch size"), _fig,
               mo.ui.table(lr_slopes, selection=None)])
    return best_refined, lr_slopes


@app.cell
def _(agg_all, mo, pd, plt):
    # Sensitivity: how many grid LRs land within SENS_TOL of the best mean RMSE for that batch size.
    SENS_TOL = 0.05
    _rows = []
    for (_opt, _bs), _g in agg_all.groupby(["optimizer", "batch_size"]):
        _m = _g["best_val_rmse"].min()
        _rows.append(dict(optimizer=_opt, batch_size=_bs,
                          n_lrs_within_tol=int((_g["best_val_rmse"] <= _m * (1 + SENS_TOL)).sum()),
                          n_lrs_tested=len(_g)))
    _sens = pd.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _opt, _g in _sens.groupby("optimizer"):
        _ax.plot(_g["batch_size"], _g["n_lrs_within_tol"], "o-", label=_opt)
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("batch size")
    _ax.set_ylabel(f"# LRs within {SENS_TOL:.0%} of best (seed mean)")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md("## Learning-rate sensitivity (wider = more forgiving)"), _fig])
    return


@app.cell
def _(mo):
    target_tol = mo.ui.slider(0.01, 0.20, step=0.01, value=0.05,
                              label="Target: within this fraction of the optimizer's best mean RMSE")
    target_tol
    return (target_tol,)


@app.cell
def _(RESULTS, SEEDS, best_mean, grid_runs, mo, pd, plt, read_history, target_tol):
    # Efficiency: optimizer steps (device-independent) to first reach a target val RMSE,
    # at the seed-mean best LR; median over the seeds that reach it.
    _lookup = {(r["optimizer"], r["batch_size"], r["lr"], r["seed"]): r for r in grid_runs}
    _rows = []
    for _opt, _g in best_mean.groupby("optimizer"):
        _target = _g["best_val_rmse"].min() * (1 + target_tol.value)
        for _rec in _g.to_dict("records"):
            _steps, _samples = [], []
            for _s in SEEDS:
                _run = _lookup[(_opt, _rec["batch_size"], _rec["lr"], _s)]
                _hit = next((h for h in read_history(RESULTS / _run["stage"], _run["run_id"])
                             if h["val_rmse"] <= _target), None)
                if _hit:
                    _steps.append(_hit["steps"])
                    _samples.append(_hit["samples_seen"])
            _rows.append(dict(optimizer=_opt, batch_size=_rec["batch_size"], target_rmse=_target,
                              seeds_reaching=f"{len(_steps)} of {len(SEEDS)}",
                              median_steps=pd.Series(_steps).median() if _steps else None,
                              median_samples=pd.Series(_samples).median() if _samples else None))
    _eff = pd.DataFrame(_rows)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _opt, _g in _eff.dropna().groupby("optimizer"):
        _axes[0].plot(_g["batch_size"], _g["median_steps"], "o-", label=_opt)
        _axes[1].plot(_g["batch_size"], _g["median_samples"], "o-", label=_opt)
    for _ax, _lab in zip(_axes, ["optimizer steps", "training samples seen"]):
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log")
        _ax.set_xlabel("batch size")
        _ax.set_ylabel(f"median {_lab} to reach target")
        _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md("## Steps and samples to reach the target RMSE (seed-mean best LR)"),
               _fig, mo.ui.table(_eff, selection=None)])
    return


@app.cell
def _(SEEDS, best_mean, grid_runs, mo, pd, plt):
    # Implicit regularization check: train-validation gap at the best epoch, mean over seeds.
    _df = pd.DataFrame(grid_runs)
    _df = _df.merge(best_mean[["optimizer", "batch_size", "lr"]], on=["optimizer", "batch_size", "lr"])
    _df["gap"] = _df["best_val_rmse"] - _df["train_rmse_at_best"]
    _gap = _df.groupby(["optimizer", "batch_size"])["gap"].agg(["mean", "std"]).reset_index()
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _opt, _g in _gap.groupby("optimizer"):
        _ax.errorbar(_g["batch_size"], _g["mean"], yerr=_g["std"], marker="o", capsize=3, label=_opt)
    _ax.axvline(32, ls="--", color="gray")
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("batch size")
    _ax.set_ylabel("val RMSE - train RMSE at best epoch")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md(f"## Generalization gap at the best epoch (mean ± std over {len(SEEDS)} seeds)"), _fig])
    return


@app.cell
def _(RESULTS, best_mean, grid_runs, mo, plt, read_history):
    # Learning curves at the seed-mean best LR, for the lowest-numbered seed.
    _seed = min(r["seed"] for r in grid_runs)
    _lookup = {(r["optimizer"], r["batch_size"], r["lr"], r["seed"]): r for r in grid_runs}
    _opts = sorted(best_mean["optimizer"].unique())
    _fig, _axes = plt.subplots(1, len(_opts), figsize=(6.5 * len(_opts), 4.5), squeeze=False)
    for _ax, _opt in zip(_axes[0], _opts):
        for _rec in best_mean[best_mean["optimizer"] == _opt].to_dict("records"):
            _run = _lookup[(_opt, _rec["batch_size"], _rec["lr"], _seed)]
            _hist = read_history(RESULTS / _run["stage"], _run["run_id"])
            _ax.plot([h["epoch"] for h in _hist], [h["val_rmse"] for h in _hist],
                     label=f"bs {_rec['batch_size']}")
        _ax.set_xscale("symlog", linthresh=10)
        _ax.set_ylim(top=1.0)
        _ax.set_xlabel("epoch")
        _ax.set_ylabel("validation RMSE")
        _ax.set_title(f"{_opt}: learning curves at best LR (seed {_seed})")
        _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    mo.vstack([mo.md("## Learning curves"), _fig])
    return


@app.cell
def _(CFG, RESULTS, SEEDS, TIE, best_mean, grid_runs, mean_over_seeds, mo, pd, read_runs, select_best):
    # Sensitivity check: do the largest batch sizes do better with a higher epoch cap?
    # capcheck re-ran every capped run at batch sizes >= min_batch_size with a higher cap;
    # runs that stopped early are unaffected by the cap, so we swap in the re-runs and
    # repeat the seed-mean selection.
    _cc = {r["run_id"]: r for r in read_runs(RESULTS / "capcheck")}
    mo.stop(not _cc, mo.md("## Epoch-cap sensitivity check\nNot run yet: `python run_sweep.py --stage capcheck`."))
    _min_bs = CFG["capcheck"]["min_batch_size"]
    _swapped = [_cc.get(r["run_id"], r) for r in grid_runs]
    _new = select_best(mean_over_seeds(_swapped, SEEDS), TIE)
    _rows = []
    for _rec in best_mean[best_mean["batch_size"] >= _min_bs].to_dict("records"):
        _n = _new[(_rec["optimizer"], _rec["batch_size"])]
        _rows.append(dict(optimizer=_rec["optimizer"], batch_size=_rec["batch_size"],
                          lr_cap1000=_rec["lr"], rmse_cap1000=_rec["best_val_rmse"],
                          lr_higher_cap=_n["lr"], rmse_higher_cap=_n["best_val_rmse"],
                          change_pct=100 * (_n["best_val_rmse"] / _rec["best_val_rmse"] - 1)))
    _still = sum(r["hit_cap"] for r in _cc.values())
    mo.vstack([mo.md(f"## Epoch-cap sensitivity check (cap {CFG['training']['max_epochs']} → "
                     f"{CFG['capcheck']['max_epochs']} epochs, batch sizes ≥ {_min_bs})\n"
                     f"{len(_cc)} capped runs re-run; {_still} still hit the higher cap. "
                     "Negative change = the larger batch size was held back by the cap."),
               mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return


@app.cell
def _(SEEDS, best_mean, grid_runs, mo, pd, plt):
    # Training speed at each batch size's best LR (seed mean): median over seeds of
    # optimizer steps (hardware-independent) and CPU seconds of training until the best epoch.
    # CPU times: one thread per run with ~11 runs in parallel, so they are indicative, not benchmarks.
    mo.stop(not SEEDS)
    _df = pd.DataFrame(grid_runs).merge(best_mean[["optimizer", "batch_size", "lr"]],
                                        on=["optimizer", "batch_size", "lr"])
    speed = (_df.groupby(["optimizer", "batch_size", "lr"])
             .agg(mean_rmse=("best_val_rmse", "mean"), median_best_epoch=("best_epoch", "median"),
                  median_steps_to_best=("steps_to_best", "median"),
                  median_cpu_sec_to_best=("train_time_to_best_s", "median"))
             .reset_index())
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _opt, _g in speed.groupby("optimizer"):
        _axes[0].plot(_g["batch_size"], _g["median_cpu_sec_to_best"], "o-", label=_opt)
        _axes[1].plot(_g["batch_size"], _g["median_steps_to_best"], "o-", label=_opt)
    for _ax, _lab in zip(_axes, ["CPU seconds to best epoch", "optimizer steps to best epoch"]):
        _ax.set_xscale("log", base=2)
        _ax.set_yscale("log")
        _ax.set_xlabel("batch size")
        _ax.set_ylabel(f"median {_lab}")
        _ax.axvline(32, ls="--", color="gray")
        _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md(f"## Training speed at the best LR (median over {len(SEEDS)} seeds)\n"
                     "CPU times are indicative (one thread per run, ~11 runs in parallel); "
                     "optimizer steps are hardware-independent."),
               _fig, mo.ui.table(speed, selection=None)])
    return (speed,)


@app.cell
def _(loso, mo):
    _per_seed =(loso.pivot_table(index=["optimizer", "batch_size"], columns="heldout_seed", values="rmse")
                 .round(4).reset_index())
    _per_seed.columns = [str(c) for c in _per_seed.columns]
    mo.vstack([mo.md("## Per-seed detail (leave-one-seed-out RMSE)"),
               mo.ui.table(_per_seed, selection=None)])
    return


if __name__ == "__main__":
    app.run()
