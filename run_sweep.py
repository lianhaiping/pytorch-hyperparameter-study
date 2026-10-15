"""Run one stage of the batch-size study. Resumable: finished runs are skipped.

Examples:
    python run_sweep.py --stage pilot
    python run_sweep.py --stage main --workers 6
    python run_sweep.py --stage extend            # repeat until it plans 0 runs
    python run_sweep.py --stage refine            # optional
    python run_sweep.py --stage capcheck          # optional sensitivity check (higher epoch cap)
    python run_sweep.py --stage main --dry-run    # list planned runs, train nothing
"""

import argparse
import copy
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.config import load_config, settings_fingerprint
from src.data import load_train_val
from src.plan import STAGES, plan
from src.repro import environment_info, resolve_device, set_determinism
from src.results import read_runs, write_env, write_run
from src.selection import dedupe
from src.runner import init_worker, run_spec

ROOT = Path(__file__).resolve().parent

# Every stage whose runs count as study results. "seedcheck" is a legacy stage (seeds 18-21
# at seed-17-selected LRs); its runs are ordinary grid/refine points and are reused as-is.
RESULT_STAGES = ("main", "extend", "refine", "seedcheck")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--config", default=str(ROOT / "configs" / "sweep.toml"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--workers", type=int, default=1, help="parallel processes (CPU: up to physical cores)")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    p.add_argument("--seeds", type=int, nargs="+", help="only run these seeds (default: all planned)")
    return p.parse_args()


def check_settings(runs: list[dict], cfg: dict, config_path: str) -> None:
    """Refuse to reuse finished runs that were made with different settings.

    A run_id covers only optimizer, batch size, LR and seed, so changing e.g. patience or
    momentum and re-running into the same results folder would otherwise skip those runs as
    "done" and silently mix old and new settings."""
    mismatched = [r for r in runs if "settings_hash" in r
                  and r["settings_hash"] != settings_fingerprint(cfg, r["optimizer"])]
    if mismatched:
        sys.exit(f"ERROR: {len(mismatched)} finished runs in this results folder were made with "
                 f"different settings than {config_path} (e.g. {mismatched[0]['run_id']}).\n"
                 "Use a new --results-dir for the changed settings, or restore the original config.")
    legacy = sum("settings_hash" not in r for r in runs)
    if legacy:
        print(f"note: {legacy} finished runs predate settings fingerprints; "
              "assuming they were made with the current config")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.stage == "capcheck":
        # Same protocol, higher epoch cap; results live only in results/capcheck.
        cfg = copy.deepcopy(cfg)
        cfg["training"]["max_epochs"] = cfg["capcheck"]["max_epochs"]
    device = resolve_device(args.device)
    results_dir = Path(args.results_dir)
    stage_dir = results_dir / args.stage

    # A run is fully determined by its RunSpec, so a run finished in any result stage is
    # never repeated.
    prior = dedupe([r for s in RESULT_STAGES for r in read_runs(results_dir / s)])
    # pilot and capcheck results are kept separate from the study results.
    separate = args.stage in ("pilot", "capcheck")
    reusable = read_runs(stage_dir) if separate else prior
    check_settings(reusable, cfg, args.config)  # before planning: mixed settings never feed selection
    specs = plan(args.stage, cfg, prior)
    done = {r["run_id"] for r in reusable}
    todo = [s for s in specs if s.run_id not in done and (not args.seeds or s.seed in args.seeds)]
    print(f"stage={args.stage} device={device} planned={len(specs)} "
          f"already_done={len(specs) - len(todo)} to_run={len(todo)}")

    if args.dry_run:
        for s in todo:
            print(f"  {s.run_id}")
        return 0
    if not todo:
        return 0
    if device.type == "cuda" and args.workers > 1:
        warnings.warn("Several workers share one GPU; consider --workers 1 or 2.")

    # Load once in the main process so the dataset download and the split file are
    # created before any worker starts (avoids concurrent downloads/writes).
    load_train_val(cfg, ROOT)
    # Apply the same settings train_one applies in every worker before recording them, so
    # env.json describes the training runs (threads, determinism), not this coordinating
    # process's defaults. The main process does no training, so this changes no results.
    set_determinism(cfg["training"]["seed"], cfg["training"]["num_threads"])
    write_env(stage_dir, {**environment_info(device), "workers": args.workers})

    t_start = time.perf_counter()

    def report(i: int, summary: dict) -> None:
        print(f"[{i}/{len(todo)}] {summary['run_id']}: best_val_rmse={summary['best_val_rmse']:.5f} "
              f"best_epoch={summary['best_epoch']} stop={summary['stop_reason']} "
              f"({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)

    # Only the main process writes result files, so parallel workers never race on them.
    if args.workers == 1:
        init_worker(cfg, str(ROOT), args.device)
        for i, spec in enumerate(todo, 1):
            summary, history, val_pred = run_spec(spec, args.stage)
            write_run(stage_dir, summary, history, val_pred)
            report(i, summary)
    else:
        # Slowest (smallest-batch) runs first, for better load balancing.
        todo.sort(key=lambda s: s.batch_size)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(cfg, str(ROOT), args.device)) as pool:
            futures = [pool.submit(run_spec, spec, args.stage) for spec in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                summary, history, val_pred = fut.result()
                write_run(stage_dir, summary, history, val_pred)
                report(i, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
