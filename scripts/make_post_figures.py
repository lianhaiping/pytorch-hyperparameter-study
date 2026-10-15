"""Make the six summary figures (one per finding) from the analysis results.

    python scripts/make_post_figures.py

Runs analysis.py headlessly and reuses its tables (loso, loso_agg, no_retune, best_mean,
best_refined, lr_slopes, speed), so every number in a figure is exactly what the notebook reports.
Writes square 1080 x 1080 PNGs to figures/.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import analysis  # noqa: E402

OUT = ROOT / "figures"
COLORS = {"sgd": "#1f77b4", "adam": "#d95f02"}
LABELS = {"sgd": "SGD (momentum 0.9)", "adam": "Adam"}
SHORT = {"sgd": "SGD", "adam": "Adam"}
# Rule of thumb for the shared batch-size region: batch sizes whose leave-one-seed-out mean RMSE
# is within REGION_TOL of that optimizer's best, for BOTH optimizers. 2% is about the size of the
# seed-to-seed noise (SGD's median seed std is ~1.5% of RMSE), so a tighter threshold would
# split the region at points that differ only by noise.
REGION_TOL = 0.02


def new_figure(title: str, headline: str):
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.text(0.06, 0.95, title, fontsize=28, fontweight="bold", va="top")
    fig.text(0.06, 0.885, headline, fontsize=18, va="top", color="#333333")
    return fig


def footer(fig, n_seeds: int, how: str = "LR chosen on the other seeds (leave-one-seed-out)") -> None:
    fig.text(0.06, 0.025, f"PyTorch MLP · California Housing · validation RMSE · {n_seeds} seeds, {how}",
             fontsize=12, color="#666666")


def bs_axis(ax, batch_sizes) -> None:
    ticks = sorted(batch_sizes)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("batch size", fontsize=16)
    ax.grid(alpha=0.3)


def add_table(fig, df: pd.DataFrame, rect) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=list(df.columns), loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 2.0)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")


def fig1_lr_scaling(d, n_seeds) -> Path:
    best, slopes = d["best_refined"], d["lr_slopes"].set_index("optimizer")  # grid + refine LRs
    s = {o: slopes.loc[o, "per_seed_le512"] for o in slopes.index}
    fig = new_figure("1. Retune the LR when batch size changes",
                     f"Best-LR slope vs batch size (≤512): SGD {s.get('sgd', '?')}, Adam {s.get('adam', '?')}")
    ax = fig.add_axes([0.11, 0.30, 0.83, 0.50])
    for opt, g in best.groupby("optimizer"):
        ax.plot(g["batch_size"], g["lr"], "o-", lw=3, ms=9, color=COLORS[opt], label=LABELS[opt])
    x = np.array([best["batch_size"].min(), best["batch_size"].max()], dtype=float)
    y0 = best["lr"].min()
    ax.plot(x, y0 * x / x[0], ":", color="gray", lw=2, label="slope 1: linear scaling rule")
    ax.plot(x, y0 * np.sqrt(x / x[0]), "--", color="gray", lw=2, alpha=0.7, label="slope 0.5: square-root rule")
    ax.set_yscale("log")
    ax.set_ylim(y0 / 3, best["lr"].max() * 3)
    ax.set_ylabel("best learning rate (mean over 5 seeds)", fontsize=16)
    bs_axis(ax, best["batch_size"].unique())
    ax.legend(fontsize=13, loc="upper left")
    tbl = slopes.reset_index()[["optimizer", "per_seed_all", "per_seed_le512"]]
    tbl["optimizer"] = tbl["optimizer"].map(LABELS)
    tbl.columns = ["optimizer", "slope, all batch sizes", "slope, batch ≤ 512"]
    add_table(fig, tbl, [0.11, 0.08, 0.83, 0.14])
    footer(fig, n_seeds, "best LR on the seed mean (grid + refine); ± = across seeds")
    return save(fig, "post_1_retune_learning_rate.png")


def fig2_no_retune(d, n_seeds) -> Path:
    nr = d["no_retune"]
    worst = nr.groupby("optimizer")["cost_pct"].max()
    fig = new_figure("2. Adam is forgiving; SGD needs care",
                     f"Batch-32 LR reused everywhere: Adam ≤ {worst.get('adam', np.nan):.1f}% worse; "
                     f"SGD up to {worst.get('sgd', np.nan):.0f}% worse")
    ax = fig.add_axes([0.11, 0.30, 0.83, 0.50])
    for opt, g in nr.groupby("optimizer"):
        ax.plot(g["batch_size"], g["cost_pct"], "o-", lw=3, ms=9, color=COLORS[opt], label=LABELS[opt])
    ax.axvline(32, ls="--", color="gray")
    ax.set_yscale("symlog", linthresh=5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(-0.5, nr["cost_pct"].max() * 1.6)
    sgd_fail = nr[(nr["optimizer"] == "sgd") & (nr["cost_pct"] > 50)]
    if len(sgd_fail):
        ax.annotate("SGD: training fails\n(small batch, LR too high)",
                    xy=(sgd_fail["batch_size"].median(), sgd_fail["cost_pct"].median()),
                    xytext=(40, -20), textcoords="offset points", fontsize=13, color=COLORS["sgd"])
    ax.set_ylabel("% worse than with a retuned LR", fontsize=16)
    bs_axis(ax, nr["batch_size"].unique())
    ax.legend(fontsize=14)
    best = d["loso_agg"].groupby("optimizer")["mean_rmse"].min()
    spread = d["loso_agg"].groupby("optimizer")["std_rmse"].median()
    tbl = pd.DataFrame({"optimizer": [LABELS[o] for o in best.index],
                        "best mean RMSE": [f"{v:.3f}" for v in best.values],
                        "median seed std": [f"{spread[o]:.4f}" for o in best.index],
                        "worst cost, fixed LR": [f"{worst[o]:.1f}%" for o in best.index]})
    add_table(fig, tbl, [0.11, 0.08, 0.83, 0.14])
    footer(fig, n_seeds)
    return save(fig, "post_2_adam_forgiving.png")


def shared_region(loso_agg: pd.DataFrame) -> list[int]:
    ok = None
    for _, g in loso_agg.groupby("optimizer"):
        good = set(g.loc[g["mean_rmse"] <= g["mean_rmse"].min() * (1 + REGION_TOL), "batch_size"])
        ok = good if ok is None else ok & good
    return sorted(ok or [])


def fig3_region(d, n_seeds) -> Path:
    agg, loso = d["loso_agg"], d["loso"]
    region = shared_region(agg)
    rtxt = f"{region[0]}–{region[-1]}" if region else "none"
    fig = new_figure("3. A region, not a magic number",
                     f"Within {REGION_TOL:.0%} of the best (≈ seed noise) for both optimizers: batch {rtxt}")
    ax = fig.add_axes([0.11, 0.30, 0.83, 0.50])
    if region:
        ax.axvspan(region[0] / 2 ** 0.4, region[-1] * 2 ** 0.4, color="#2ca02c", alpha=0.12,
                   label=f"within {REGION_TOL:.0%} of best, both optimizers ({rtxt})")
    for opt, g in agg.groupby("optimizer"):
        ax.errorbar(g["batch_size"], g["mean_rmse"], yerr=g["std_rmse"], marker="o", ms=9, lw=3,
                    capsize=4, color=COLORS[opt], label=f"{LABELS[opt]}: mean ± std")
        pts = loso[loso["optimizer"] == opt]
        ax.scatter(pts["batch_size"], pts["rmse"], s=18, alpha=0.3, color=COLORS[opt])
    ax.axvline(32, ls="--", color="gray", label="batch size 32")
    # Zoom on the curve; single-seed blow-ups are annotated instead of stretching the axis.
    y_lo = min(agg["mean_rmse"].min(), loso["rmse"].min()) - 0.005
    y_hi = agg.loc[agg["mean_rmse"] < agg["mean_rmse"].median() * 1.05, "mean_rmse"].max() + 0.02
    ax.set_ylim(y_lo, y_hi)
    for _, r in loso[loso["rmse"] > y_hi].iterrows():
        what = "diverged" if r["stop_reason"] == "diverged" else "off chart"
        ax.annotate(f"{SHORT[r['optimizer']]} @ {int(r['batch_size'])}, seed {int(r['heldout_seed'])}:\n"
                    f"{what}, RMSE {r['rmse']:.2f}", xy=(r["batch_size"], y_hi), xytext=(0, -45),
                    textcoords="offset points", ha="center", fontsize=11, color=COLORS[r["optimizer"]],
                    arrowprops=dict(arrowstyle="->", color=COLORS[r["optimizer"]]))
    ax.set_ylabel("validation RMSE ($100k)", fontsize=16)
    bs_axis(ax, agg["batch_size"].unique())
    ax.legend(fontsize=11, loc="lower center", ncol=2)
    rows = []
    for opt, g in agg.groupby("optimizer"):
        b = g.loc[g["mean_rmse"].idxmin()]
        r32 = g.loc[g["batch_size"] == 32, "mean_rmse"]
        rows.append([LABELS[opt], int(b["batch_size"]), f"{b['mean_rmse']:.3f}",
                     f"{r32.iloc[0]:.3f}" if len(r32) else "—"])
    add_table(fig, pd.DataFrame(rows, columns=["optimizer", "best batch", "best RMSE", "RMSE at 32"]),
              [0.11, 0.08, 0.83, 0.14])
    footer(fig, n_seeds)
    return save(fig, "post_3_batch_size_region.png")


def fig4_seeds(d, n_seeds) -> Path:
    loso, agg = d["loso"], d["loso_agg"]
    winners = (loso.loc[loso.groupby(["optimizer", "heldout_seed"])["rmse"].idxmin()]
               .pivot(index="optimizer", columns="heldout_seed", values="batch_size"))
    n_distinct = winners.nunique(axis=1)
    distinct_txt = " and ".join(f"{SHORT[o]} {n_distinct[o]}" for o in n_distinct.index)
    fig = new_figure("4. One seed can fool you",
                     f"Different 'best' batch sizes across {n_seeds} seeds: {distinct_txt}")

    # Table: which batch size wins on each held-out seed.
    tbl = winners.reset_index()
    tbl["optimizer"] = tbl["optimizer"].map(SHORT)
    tbl.columns = ["best batch"] + [f"seed {c}" for c in winners.columns]
    add_table(fig, tbl, [0.06, 0.66, 0.88, 0.14])

    # Bars: seed-to-seed noise vs the batch-size effect in the flat middle of the curve.
    ax = fig.add_axes([0.14, 0.14, 0.80, 0.40])
    region = shared_region(agg)
    lo, hi = (region[0], region[-1]) if region else (4, 256)
    mid = agg[(agg["batch_size"] >= lo) & (agg["batch_size"] <= hi)]
    opts = sorted(agg["optimizer"].unique())
    seed_noise = [agg[agg["optimizer"] == o]["std_rmse"].median() for o in opts]
    bs_effect = [mid[mid["optimizer"] == o]["mean_rmse"].std() for o in opts]
    x = np.arange(len(opts))
    ax.bar(x - 0.2, seed_noise, 0.38, color="#7f7f7f", label="seed-to-seed std (median over batch sizes)")
    ax.bar(x + 0.2, bs_effect, 0.38, color=[COLORS[o] for o in opts],
           label=f"std across batch sizes {lo}–{hi} (of the seed-mean RMSE)")
    for xi, (sn, be) in enumerate(zip(seed_noise, bs_effect)):
        ax.text(xi, max(sn, be) * 1.04, f"seed noise = {sn / be:.1f}× batch-size effect",
                ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(max(seed_noise), max(bs_effect)) * 1.3)
    ax.set_xticks(x, [LABELS[o] for o in opts], fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("RMSE ($100k)", fontsize=16)
    ax.set_title("Seed noise vs batch-size effect", fontsize=18)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.06, 0.075, "Differences smaller than the seed-to-seed spread are not findings.",
             fontsize=15, style="italic")
    footer(fig, n_seeds)
    return save(fig, "post_4_one_seed_can_fool_you.png")


def fig5_speed(d, n_seeds) -> Path:
    sp, agg = d["speed"], d["loso_agg"]
    region = shared_region(agg)
    lo, hi = (region[0], region[-1]) if region else (4, 256)

    def at(opt, bs, col):
        return sp.loc[(sp["optimizer"] == opt) & (sp["batch_size"] == bs), col].iloc[0]

    speedup = {o: at(o, lo, "median_cpu_sec_to_best") / at(o, hi, "median_cpu_sec_to_best")
               for o in sorted(sp["optimizer"].unique())}
    sp_txt = ", ".join(f"{SHORT[o]} {v:.0f}×" for o, v in speedup.items())
    fig = new_figure("5. Same accuracy, much faster",
                     f"Inside the region, batch {hi} trains faster than batch {lo}: {sp_txt}")
    ax = fig.add_axes([0.11, 0.30, 0.83, 0.50])
    ax.axvspan(lo / 2 ** 0.4, hi * 2 ** 0.4, color="#2ca02c", alpha=0.12,
               label=f"accuracy region ({lo}–{hi}, within {REGION_TOL:.0%} of best)")
    for opt, g in sp.groupby("optimizer"):
        ax.plot(g["batch_size"], g["median_cpu_sec_to_best"], "o-", lw=3, ms=9, color=COLORS[opt],
                label=LABELS[opt])
    ax.axvline(32, ls="--", color="gray", label="batch size 32")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_ylabel("CPU time to best model (median)", fontsize=16)
    bs_axis(ax, sp["batch_size"].unique())
    ax.legend(fontsize=12, loc="upper right")
    rows = []
    for opt in sorted(sp["optimizer"].unique()):
        cells = [SHORT[opt]]
        for bs in (lo, 32, hi):
            cells.append(f"{at(opt, bs, 'mean_rmse'):.3f} · {at(opt, bs, 'median_cpu_sec_to_best'):.0f} s")
        rows.append(cells)
    add_table(fig, pd.DataFrame(rows, columns=["RMSE · time", f"batch {lo}", "batch 32", f"batch {hi}"]),
              [0.11, 0.08, 0.83, 0.14])
    footer(fig, n_seeds, "CPU time, 1 thread per run (indicative)")
    return save(fig, "post_5_same_accuracy_faster.png")


def fig6_head_to_head(d, n_seeds, small: int = 32, large: int = 256) -> Path:
    loso, agg, sp = d["loso"], d["loso_agg"], d["speed"]
    opts = sorted(loso["optimizer"].unique())

    def sp_at(opt, bs, col):
        return sp.loc[(sp["optimizer"] == opt) & (sp["batch_size"] == bs), col].iloc[0]

    def rmse_at(opt, bs):
        return agg.loc[(agg["optimizer"] == opt) & (agg["batch_size"] == bs), "mean_rmse"].iloc[0]

    speedups = {o: sp_at(o, small, "median_cpu_sec_to_best") / sp_at(o, large, "median_cpu_sec_to_best")
                for o in opts}
    diffs = {o: 100 * (rmse_at(o, large) / rmse_at(o, small) - 1) for o in opts}
    fig = new_figure(f"6. Batch {large} vs batch {small}",
                     f"Same accuracy (within seed noise), {min(speedups.values()):.1f}–"
                     f"{max(speedups.values()):.1f}× faster, with the LR retuned")

    # Left: per-seed accuracy, paired by seed.
    ax1 = fig.add_axes([0.10, 0.32, 0.38, 0.46])
    for i, opt in enumerate(opts):
        piv = loso[loso["optimizer"] == opt].pivot(index="heldout_seed", columns="batch_size", values="rmse")
        x0, x1 = i * 3, i * 3 + 1
        for _, row in piv.iterrows():
            ax1.plot([x0, x1], [row[small], row[large]], "-", color=COLORS[opt], alpha=0.35, lw=1.5)
        ax1.plot([x0] * len(piv), piv[small], "o", color=COLORS[opt], ms=8)
        ax1.plot([x1] * len(piv), piv[large], "s", color=COLORS[opt], ms=8)
        ax1.plot([x0, x1], [piv[small].mean(), piv[large].mean()], "-", color=COLORS[opt], lw=4,
                 label=f"{SHORT[opt]}: mean ({diffs[opt]:+.1f}%)")
    ax1.set_xticks([i * 3 + k for i in range(len(opts)) for k in (0, 1)],
                   [f"{SHORT[o]}\n{bs}" for o in opts for bs in (small, large)], fontsize=13)
    ax1.set_xlim(-0.6, 3 * len(opts) - 1.4)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_ylabel("validation RMSE, each seed", fontsize=15)
    ax1.set_title("Accuracy (lines = same seed)", fontsize=16)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Right: time to best model.
    ax2 = fig.add_axes([0.58, 0.32, 0.37, 0.46])
    w = 0.38
    for i, opt in enumerate(opts):
        t_s, t_l = sp_at(opt, small, "median_cpu_sec_to_best"), sp_at(opt, large, "median_cpu_sec_to_best")
        ax2.bar(i - w / 2, t_s, w, color=COLORS[opt], alpha=0.45)
        ax2.bar(i + w / 2, t_l, w, color=COLORS[opt])
        ax2.text(i, max(t_s, t_l) * 1.05, f"{speedups[opt]:.1f}× faster", ha="center", fontsize=14,
                 fontweight="bold")
    ax2.set_xticks([i + k * w / 2 for i in range(len(opts)) for k in (-1, 1)],
                   [f"{SHORT[o]}\n{bs}" for o in opts for bs in (small, large)], fontsize=13)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylabel("CPU seconds to best model (median)", fontsize=15)
    ax2.set_title("Training time", fontsize=16)
    ax2.set_ylim(0, max(sp_at(o, small, "median_cpu_sec_to_best") for o in opts) * 1.25)
    ax2.grid(axis="y", alpha=0.3)

    rows = []
    for opt in opts:
        rows.append([SHORT[opt],
                     f"{sp_at(opt, small, 'lr'):g} → {sp_at(opt, large, 'lr'):g}",
                     f"{rmse_at(opt, small):.3f} → {rmse_at(opt, large):.3f}",
                     f"{sp_at(opt, small, 'median_cpu_sec_to_best'):.0f} s → "
                     f"{sp_at(opt, large, 'median_cpu_sec_to_best'):.0f} s"])
    add_table(fig, pd.DataFrame(rows, columns=["", f"LR ({small}→{large})",
                                               "mean RMSE", "time to best"]), [0.06, 0.08, 0.88, 0.14])
    footer(fig, n_seeds)
    return save(fig, f"post_6_batch_{large}_vs_{small}.png")


def save(fig, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> int:
    _, d = analysis.app.run()
    missing = [k for k in ("loso", "loso_agg", "no_retune", "best_mean", "best_refined", "lr_slopes", "speed")
               if k not in d]
    if missing:
        print(f"analysis did not produce {missing}; are at least 2 seeds complete?")
        return 1
    n_seeds = len(d["SEEDS"])
    for f in (fig1_lr_scaling, fig2_no_retune, fig3_region, fig4_seeds, fig5_speed, fig6_head_to_head):
        print("wrote", f(d, n_seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
