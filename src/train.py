"""Training: fit one (optimizer, batch size, learning rate, seed) configuration.

Evaluation (metrics, predictions) lives in evaluate.py; inference from a saved model is
shown in scripts/inference_example.py."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import TrainValData
from .evaluate import predict, rmse
from .model import build_model
from .config import settings_fingerprint
from .repro import set_determinism, sync


@dataclass(frozen=True)
class RunSpec:
    optimizer: str
    batch_size: int
    lr: float
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.optimizer}_bs{self.batch_size}_lr{self.lr:.4g}_s{self.seed}"


def make_optimizer(name: str, params, lr: float, opt_cfg: dict) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=opt_cfg["momentum"],
                               nesterov=opt_cfg["nesterov"], weight_decay=opt_cfg["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(opt_cfg["betas"]),
                                eps=opt_cfg["eps"], weight_decay=opt_cfg["weight_decay"])
    raise ValueError(f"unknown optimizer '{name}'")


def train_one(spec: RunSpec, data: TrainValData, cfg: dict, device: torch.device, stage: str,
              return_state: bool = False):
    """Returns (summary dict, per-epoch history list, best-model validation predictions),
    plus the best-epoch state_dict (on CPU) when return_state=True."""
    tc = cfg["training"]
    set_determinism(spec.seed, tc["num_threads"])

    X_train = torch.from_numpy(data.X_train).to(device)
    y_train = torch.from_numpy(data.y_train).to(device)
    X_val = torch.from_numpy(data.X_val).to(device)
    y_val = torch.from_numpy(data.y_val).to(device)
    n_train, in_features = X_train.shape
    bs = spec.batch_size

    model = build_model(spec.seed, in_features, cfg["model"]["hidden_sizes"], device)
    # Created after model.to(device) so optimizer state lives on the same device.
    optimizer = make_optimizer(spec.optimizer, model.parameters(), spec.lr, cfg["optimizers"][spec.optimizer])
    loss_fn = nn.MSELoss()
    # Shuffle order comes from a dedicated CPU generator, so it is identical on every device.
    shuffle_gen = torch.Generator(device="cpu").manual_seed(spec.seed)

    # Evaluation (evaluate.rmse) switches to model.eval() and runs under torch.no_grad().
    initial_train_rmse = rmse(model, X_train, y_train)
    initial_val_rmse = rmse(model, X_val, y_val)

    history = [dict(epoch=0, steps=0, samples_seen=0, train_time_s=0.0,
                    train_rmse=initial_train_rmse, val_rmse=initial_val_rmse)]
    steps = samples_seen = 0
    train_time = 0.0

    best_val = initial_val_rmse
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stop_reason = "max_epochs"

    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()  # back to training mode after the previous epoch's evaluation
        sync(device)
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=shuffle_gen).to(device)
        nonfinite_loss = False
        for start in range(0, n_train, bs):
            idx = perm[start:start + bs]
            if tc["drop_last"] and idx.numel() < bs:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            if not torch.isfinite(loss):
                nonfinite_loss = True
                break
            loss.backward()
            optimizer.step()
            steps += 1
            samples_seen += idx.numel()
        sync(device)
        train_time += time.perf_counter() - t0

        if nonfinite_loss:
            history.append(dict(epoch=epoch, steps=steps, samples_seen=samples_seen,
                                train_time_s=train_time, train_rmse=math.nan, val_rmse=math.nan))
            stop_reason = "diverged"
            break

        train_rmse = rmse(model, X_train, y_train)  # eval mode, no gradients
        val_rmse = rmse(model, X_val, y_val)
        history.append(dict(epoch=epoch, steps=steps, samples_seen=samples_seen,
                            train_time_s=train_time, train_rmse=train_rmse, val_rmse=val_rmse))

        if not math.isfinite(val_rmse) or val_rmse > tc["divergence_factor"] * initial_val_rmse:
            stop_reason = "diverged"
            break
        if val_rmse < best_val - tc["min_delta"]:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= tc["patience"]:
            stop_reason = "patience"
            break

    # Restore the best-epoch weights and re-derive the validation RMSE from them.
    model.load_state_dict(best_state)
    val_pred = predict(model, X_val)  # eval mode, no gradients
    check = float(np.sqrt(np.mean((val_pred.astype(np.float64) - data.y_val.astype(np.float64)) ** 2)))
    assert math.isclose(check, best_val, rel_tol=1e-5), f"restored-model RMSE {check} != recorded {best_val}"

    best_row = history[best_epoch]
    assert best_row["epoch"] == best_epoch
    last_row = history[-1]
    steps_per_epoch = n_train // bs if tc["drop_last"] else math.ceil(n_train / bs)

    summary = dict(
        run_id=spec.run_id,
        stage=stage,
        optimizer=spec.optimizer,
        batch_size=bs,
        lr=spec.lr,
        seed=spec.seed,
        best_val_rmse=best_val,
        train_rmse_at_best=best_row["train_rmse"],
        best_epoch=best_epoch,
        epochs_run=last_row["epoch"],
        stop_reason=stop_reason,
        hit_cap=stop_reason == "max_epochs",
        diverged=stop_reason == "diverged",
        initial_val_rmse=initial_val_rmse,
        steps_to_best=best_row["steps"],
        samples_to_best=best_row["samples_seen"],
        train_time_to_best_s=best_row["train_time_s"],
        total_steps=last_row["steps"],
        total_train_time_s=last_row["train_time_s"],
        n_train=n_train,
        steps_per_epoch=steps_per_epoch,
        device=str(device),
        # Hash of the settings outside the run_id; run_sweep.py refuses to mix fingerprints.
        settings_hash=settings_fingerprint(cfg, spec.optimizer),
    )
    if return_state:
        return summary, history, val_pred.ravel().astype(np.float32), best_state
    return summary, history, val_pred.ravel().astype(np.float32)
