"""One-image summary of the study: hook, setup, and six findings, each with a mini chart.
1080 x 1350 (LinkedIn 4:5).

    python scripts/make_summary_card.py

Every number and chart is computed from analysis.py's own tables (same source as
make_post_figures.py).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import analysis  # noqa: E402
from make_post_figures import COLORS, OUT, REGION_TOL, SHORT, shared_region  # noqa: E402

INK, MUTED, ACCENT, STAT = "#1a1a1a", "#555555", "#2ca02c", "#1f4e79"
OPTS = ("adam", "sgd")


def numbers(d) -> dict:
    agg, sp, nr, slopes = d["loso_agg"], d["speed"], d["no_retune"], d["lr_slopes"].set_index("optimizer")
    region = shared_region(agg)
    lo, hi = region[0], region[-1]

    def t(opt, bs):
        return sp.loc[(sp["optimizer"] == opt) & (sp["batch_size"] == bs), "median_cpu_sec_to_best"].iloc[0]

    def r(opt, bs):
        return agg.loc[(agg["optimizer"] == opt) & (agg["batch_size"] == bs), "mean_rmse"].iloc[0]

    mid = agg[(agg["batch_size"] >= lo) & (agg["batch_size"] <= hi)]
    seed_noise = {o: agg[agg["optimizer"] == o]["std_rmse"].median() for o in OPTS}
    bs_effect = {o: mid[mid["optimizer"] == o]["mean_rmse"].std() for o in OPTS}
    worst = nr.groupby("optimizer")["cost_pct"].max()
    return dict(
        n_runs=len(d["runs"]), n_seeds=len(d["SEEDS"]), lo=lo, hi=hi,
        slope_sgd=slopes.loc["sgd", "per_seed_le512"], slope_adam=slopes.loc["adam", "per_seed_le512"],
        worst_adam=worst["adam"], worst_sgd=worst["sgd"],
        seed_noise=seed_noise, bs_effect=bs_effect,
        seed_ratio={o: seed_noise[o] / bs_effect[o] for o in OPTS},
        speed=dict(sgd=t("sgd", lo) / t("sgd", hi), adam=t("adam", lo) / t("adam", hi)),
        t32={o: t(o, 32) for o in OPTS}, t256={o: t(o, 256) for o in OPTS},
        d256={o: 100 * (r(o, 256) / r(o, 32) - 1) for o in OPTS},
    )


def style_mini(ax, log2x=True) -> None:
    if log2x:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 8, 32, 128, 512, 2048, 8192], ["2", "8", "32", "128", "512", "2k", "8k"])
        ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=9, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)


def card(fig, x, y, w, h, num, title, stat, explain):
    """Draws the card frame, number, title, key stat and explanation; returns the mini-chart rect."""
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor="#f4f6f8", edgecolor="#d0d5da",
                                      zorder=-1))  # behind the mini chart
    fig.text(x + 0.015, y + h - 0.012, num, fontsize=20, fontweight="bold", color=ACCENT, va="top")
    fig.text(x + 0.052, y + h - 0.015, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(x + 0.052, y + h - 0.042, stat, fontsize=11.5, fontweight="bold", color=STAT, va="top")
    fig.text(x + 0.015, y + 0.009, explain, fontsize=9.2, color=MUTED, va="bottom", linespacing=1.3)
    return [x + 0.065, y + 0.075, w - 0.085, h - 0.14]


def main() -> int:
    _, d = analysis.app.run()
    n = numbers(d)
    agg, sp, nr, best = d["loso_agg"], d["speed"], d["no_retune"], d["best_refined"]
    lo, hi = n["lo"], n["hi"]

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.patch.set_facecolor("white")

    # Hook + study summary
    fig.text(0.05, 0.972, "Batch size 32 isn't gold. The learning rate is.", fontsize=14, color=MUTED, va="top",
             style="italic")
    fig.text(0.05, 0.947, "Friends don't let friends choose a batch size\nwithout checking the learning rate "
             "and the run time.", fontsize=21, fontweight="bold", color=INK, va="top", linespacing=1.2)
    fig.text(0.05, 0.868,
             f"A miniature, reproducible study: {n['n_runs']:,} training runs · PyTorch MLP · California Housing · "
             f"batch 2–8,192 ·\nSGD & Adam · 9 learning rates · {n['n_seeds']} seeds · one desktop CPU",
             fontsize=10.5, color=MUTED, va="top", linespacing=1.35)
    # Shared legend for the mini charts
    for i, o in enumerate(OPTS):
        fig.text(0.05 + i * 0.1, 0.832, f"━ {SHORT[o]}", fontsize=12, fontweight="bold", color=COLORS[o], va="top")
    fig.text(0.26, 0.832, f"green band = batch {lo}–{hi} (within {REGION_TOL:.0%} of best)", fontsize=11,
             color=ACCENT, va="top")

    w, h, gx, gy, x0, top = 0.44, 0.232, 0.02, 0.013, 0.05, 0.808

    def rect(i):
        col, row = i % 2, i // 2
        return x0 + col * (w + gx), top - (row + 1) * h - row * gy

    # 1. Best LR vs batch size
    r = card(fig, *rect(0), w, h, "1", "Retune the LR with batch size",
             f"SGD slope {n['slope_sgd']} · Adam {n['slope_adam']}",
             "Best learning rate at each batch size (mean over 5 seeds).\n"
             "SGD's rises about 1:1 with batch size (linear rule) up to\n"
             "~1k; Adam's rises more gently.")
    ax = fig.add_axes(r)
    for o in OPTS:
        g = best[best["optimizer"] == o]
        ax.plot(g["batch_size"], g["lr"], "o-", ms=3.5, lw=2, color=COLORS[o])
    xs = np.array([2, 8192.0])
    y0 = best["lr"].min()
    ax.plot(xs, y0 * xs / 2, ":", color="gray", lw=1.5)
    ax.text(700, y0 * 700 / 2 * 1.6, "slope 1", fontsize=8.5, color="gray", rotation=28)
    ax.set_yscale("log")
    ax.set_ylim(y0 / 2, best["lr"].max() * 3)
    ax.set_ylabel("best LR", fontsize=9.5)
    style_mini(ax)

    # 2. Cost of not retuning
    r = card(fig, *rect(1), w, h, "2", "Adam is forgiving; SGD needs care",
             f"Fixed LR: Adam ≤{n['worst_adam']:.1f}%, SGD ≤{n['worst_sgd']:.0f}% worse",
             "What happens if the LR tuned at batch 32 is reused\n"
             "everywhere: Adam barely notices; SGD fails to train at\n"
             "small batches and degrades at large ones.")
    ax = fig.add_axes(r)
    for o in OPTS:
        g = nr[nr["optimizer"] == o]
        ax.plot(g["batch_size"], g["cost_pct"], "o-", ms=3.5, lw=2, color=COLORS[o])
    ax.set_yscale("symlog", linthresh=5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(-0.5, nr["cost_pct"].max() * 1.8)
    ax.set_ylabel("% worse", fontsize=9.5)
    style_mini(ax)

    # 3. Tuned RMSE vs batch size with region
    r = card(fig, *rect(2), w, h, "3", "A region, not a magic number",
             f"Batch {lo}–{hi} within {REGION_TOL:.0%} of the best",
             "Tuned accuracy (LR chosen on 4 seeds, scored on the 5th).\n"
             f"Everything in the band is within {REGION_TOL:.0%} of the best,\n"
             "about the size of seed noise. 32 is inside, not special.")
    ax = fig.add_axes(r)
    ax.axvspan(lo / 2 ** 0.4, hi * 2 ** 0.4, color=ACCENT, alpha=0.13)
    # y-range from the typical points. A point far above (one diverged seed pulling the mean up)
    # is drawn at the top edge with a triangle marker and labelled with its true value, so the
    # line stays connected and nothing runs off the chart.
    med = agg["mean_rmse"].median()
    normal = agg[agg["mean_rmse"] < med * 1.05]
    ylo, yhi = normal["mean_rmse"].min() - 0.003, normal["mean_rmse"].max() + 0.006
    y_cap = yhi - 0.0012
    for o in OPTS:
        g = agg[agg["optimizer"] == o].sort_values("batch_size")
        y = g["mean_rmse"].clip(upper=y_cap)
        ax.plot(g["batch_size"], y, "o-", ms=3.5, lw=2, color=COLORS[o])
    ax.axvline(32, ls="--", color="gray", lw=1)
    ax.set_ylim(ylo, yhi)
    for _, row in agg[agg["mean_rmse"] > y_cap].iterrows():
        c = COLORS[row["optimizer"]]
        ax.plot(row["batch_size"], y_cap, "^", ms=8, color=c)
        ax.text(row["batch_size"] * 1.3, y_cap,
                f"off scale: mean {row['mean_rmse']:.2f}\n(1 seed diverged)",
                fontsize=7.5, color=c, va="top")
    ax.set_ylabel("val RMSE", fontsize=9.5)
    style_mini(ax)

    # 4. Seed noise vs batch-size effect
    ratio = n["seed_ratio"]
    r = card(fig, *rect(3), w, h, "4", "One seed can fool you",
             f"Seed noise = {min(ratio.values()):.0f}–{max(ratio.values()):.0f}× the batch-size effect",
             "Grey: typical seed-to-seed spread (init + batch order).\n"
             f"Colour: spread across batch sizes {lo}–{hi}. The seed moved\n"
             "results more than the batch size did.")
    ax = fig.add_axes(r)
    xpos = np.arange(len(OPTS))
    ax.bar(xpos - 0.2, [n["seed_noise"][o] for o in OPTS], 0.38, color="#8c8c8c", label="seed-to-seed std")
    ax.bar(xpos + 0.2, [n["bs_effect"][o] for o in OPTS], 0.38, color=[COLORS[o] for o in OPTS],
           label=f"batch-size effect ({lo}–{hi})")
    ax.set_xticks(xpos, [SHORT[o] for o in OPTS], fontsize=10)
    ax.set_ylabel("RMSE spread", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    style_mini(ax, log2x=False)

    # 5. Time to best vs batch size
    r = card(fig, *rect(4), w, h, "5", "Same accuracy, much faster",
             f"{hi} vs {lo}: {n['speed']['sgd']:.0f}× (SGD), {n['speed']['adam']:.0f}× (Adam) faster",
             "CPU time to reach the best model. Inside the band the\n"
             f"accuracy is the same, but batch {hi} gets there in seconds\n"
             f"instead of minutes.")
    ax = fig.add_axes(r)
    ax.axvspan(lo / 2 ** 0.4, hi * 2 ** 0.4, color=ACCENT, alpha=0.13)
    for o in OPTS:
        g = sp[sp["optimizer"] == o]
        ax.plot(g["batch_size"], g["median_cpu_sec_to_best"], "o-", ms=3.5, lw=2, color=COLORS[o])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}s"))
    ax.set_ylabel("CPU time to best", fontsize=9.5)
    style_mini(ax)

    # 6. Batch 256 vs 32
    sp6 = {o: n["t32"][o] / n["t256"][o] for o in OPTS}

    def _lr(o, bs):
        return sp.loc[(sp["optimizer"] == o) & (sp["batch_size"] == bs), "lr"].iloc[0]

    lr6 = ", ".join(f"{SHORT[o]} {_lr(o, 32):g}→{_lr(o, 256):g}" for o in OPTS)
    r = card(fig, *rect(5), w, h, "6", "Batch 256 vs 32, head to head",
             f"Same accuracy, {min(sp6.values()):.1f}–{max(sp6.values()):.1f}× faster",
             f"LR retuned ({lr6}).\n"
             f"Batch 256 matched batch 32 within seed noise\n"
             f"({min(n['d256'].values()):+.1f} to {max(n['d256'].values()):+.1f}% RMSE) "
             "and trained about 4× faster.")
    ax = fig.add_axes(r)
    for i, o in enumerate(OPTS):
        ax.bar(i - 0.2, n["t32"][o], 0.38, color=COLORS[o], alpha=0.4)
        ax.bar(i + 0.2, n["t256"][o], 0.38, color=COLORS[o])
        ax.text(i, n["t32"][o] * 1.06, f"{sp6[o]:.1f}×", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([i + k for i in range(len(OPTS)) for k in (-0.2, 0.2)],
                  [f"{SHORT[o]} {bs}" for o in OPTS for bs in (32, 256)], fontsize=8.5)
    ax.set_ylim(0, max(n["t32"].values()) * 1.3)
    ax.set_ylabel("CPU s to best", fontsize=9.5)
    style_mini(ax, log2x=False)

    # Takeaway + footer
    fig.text(0.5, 0.062, "What mattered most:  learning rate  >  seed  >  batch size", fontsize=18,
             fontweight="bold", color=INK, ha="center")
    fig.text(0.5, 0.022, "Validation RMSE · LR chosen on 4 seeds, scored on the 5th · constant LR, no warmup · "
             "CPU time, one thread per run\nSmall tabular MLP: the numbers are setting-specific; the principles "
             "are backed by larger-scale research · code & raw results on GitHub",
             fontsize=9.5, color=MUTED, ha="center", linespacing=1.35)

    OUT.mkdir(exist_ok=True)
    path = OUT / "post_0_summary_card.png"
    fig.savefig(path, dpi=100, facecolor="white")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
