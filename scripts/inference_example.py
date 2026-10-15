"""Example (not part of the study): train one configuration, save it, load it back, and
run inference on raw, unscaled inputs.

    python scripts/inference_example.py --optimizer adam --batch-size 64 --lr 3e-3

Shows the full lifecycle: train -> save checkpoint (weights + scaler) -> load into a fresh
model -> model.eval() + torch.no_grad() -> predict. Predictions are made on validation rows
(the test split is never used), and are checked against the predictions made during training.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import N_SAMPLES, load_or_create_split, load_train_val  # noqa: E402
# Imported under a distinct name so it isn't confused with the Predictor.predict method below.
from src.evaluate import predict as model_predict  # noqa: E402
from src.model import MLP  # noqa: E402
from src.repro import resolve_device  # noqa: E402
from src.train import RunSpec, train_one  # noqa: E402


def save_checkpoint(path: Path, state_dict: dict, hidden_sizes: list[int], scaler_mean, scaler_scale,
                    spec: RunSpec) -> None:
    # Everything is stored as tensors / plain Python types so torch.load(weights_only=True) works.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state_dict,
        "hidden_sizes": list(hidden_sizes),
        "scaler_mean": torch.as_tensor(scaler_mean, dtype=torch.float64),
        "scaler_scale": torch.as_tensor(scaler_scale, dtype=torch.float64),
        "spec": dict(optimizer=spec.optimizer, batch_size=spec.batch_size, lr=spec.lr, seed=spec.seed),
    }, path)


class Predictor:
    """Loads a checkpoint and predicts median house value ($100k) from raw feature rows."""

    def __init__(self, path: Path, device: torch.device):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.device = device
        self.mean = ckpt["scaler_mean"].to(device)
        self.scale = ckpt["scaler_scale"].to(device)
        self.model = MLP(in_features=len(self.mean), hidden_sizes=tuple(ckpt["hidden_sizes"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()  # inference mode for layers like dropout/BatchNorm (none here, but always set it)

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X = torch.as_tensor(X_raw, dtype=torch.float64, device=self.device)
        X = ((X - self.mean) / self.scale).float()  # same standardization as training
        return model_predict(self.model, X).ravel()  # eval mode + torch.no_grad(), back on the CPU


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "sweep.toml"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--optimizer", default="adam", choices=("sgd", "adam"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--epochs", type=int, default=None, help="optional max_epochs override")
    p.add_argument("--out", default=str(ROOT / "models" / "example.pt"))
    args = p.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    device = resolve_device(args.device)
    data = load_train_val(cfg, ROOT)
    spec = RunSpec(args.optimizer, args.batch_size, args.lr, cfg["training"]["seed"])

    # 1. Train (early stopping restores the best-epoch weights).
    summary, _, val_pred_train, state = train_one(spec, data, cfg, device, stage="example", return_state=True)
    print(f"trained {spec.run_id}: best val RMSE {summary['best_val_rmse']:.4f} at epoch {summary['best_epoch']}")

    # 2. Save weights + preprocessing together: a model is useless without its scaler.
    out = Path(args.out)
    save_checkpoint(out, state, cfg["model"]["hidden_sizes"], data.scaler_mean, data.scaler_scale, spec)
    print(f"saved checkpoint to {out}")

    # 3. Load into a fresh model and predict on RAW validation rows.
    X, y = fetch_california_housing(return_X_y=True)
    dc = cfg["data"]
    split = load_or_create_split(ROOT / dc["splits_file"], N_SAMPLES, dc["val_size"], dc["test_size"], dc["split_seed"])
    predictor = Predictor(out, device)
    pred = predictor.predict(X[split.val_idx])

    rmse = float(np.sqrt(np.mean((pred - y[split.val_idx]) ** 2)))
    same = np.allclose(pred, val_pred_train, atol=1e-4)
    print(f"reloaded model: val RMSE {rmse:.4f}; matches training-time predictions: {same}")
    for i in range(3):
        print(f"  row {split.val_idx[i]}: predicted ${pred[i] * 100_000:,.0f}  actual ${y[split.val_idx[i]] * 100_000:,.0f}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
