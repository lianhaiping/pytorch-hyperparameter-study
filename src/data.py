"""California Housing: frozen train/val/test split and train-only standardization.

The test split is created so that its indices are frozen and committed with the
repo, but this module never returns test rows: the study uses train and
validation only.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 20640
N_FEATURES = 8


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class TrainValData:
    X_train: np.ndarray  # float32, (n_train, 8), standardized with train statistics
    y_train: np.ndarray  # float32, (n_train, 1), raw target in units of $100k
    X_val: np.ndarray    # float32, (n_val, 8), standardized with train statistics
    y_val: np.ndarray    # float32, (n_val, 1)
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def make_split(n: int, val_size: int, test_size: int, seed: int) -> Split:
    """Two-step split: carve off test first, then validation from the remainder."""
    all_idx = np.arange(n)
    rest_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(rest_idx, test_size=val_size, random_state=seed)
    # Sorted so the saved file is canonical; row order is irrelevant because
    # training reshuffles every epoch.
    split = Split(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))
    check_split(split, n, val_size, test_size)
    return split


def check_split(split: Split, n: int, val_size: int, test_size: int) -> None:
    tr, va, te = split.train_idx, split.val_idx, split.test_idx
    assert len(va) == val_size and len(te) == test_size and len(tr) == n - val_size - test_size
    assert len(np.intersect1d(tr, va)) == 0, "train/val overlap"
    assert len(np.intersect1d(tr, te)) == 0, "train/test overlap"
    assert len(np.intersect1d(va, te)) == 0, "val/test overlap"
    assert np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(n)), "split does not cover all rows"


def load_or_create_split(path: Path, n: int, val_size: int, test_size: int, seed: int) -> Split:
    """The saved split file is authoritative, so results never depend on sklearn's
    splitting implementation staying the same across versions."""
    if path.exists():
        saved = np.load(path)
        meta = (int(saved["n"]), int(saved["val_size"]), int(saved["test_size"]), int(saved["split_seed"]))
        if meta != (n, val_size, test_size, seed):
            raise ValueError(f"{path} was made with (n, val, test, seed)={meta}; config asks for "
                             f"{(n, val_size, test_size, seed)}. Delete the file or fix the config.")
        split = Split(saved["train_idx"], saved["val_idx"], saved["test_idx"])
        check_split(split, n, val_size, test_size)
        recomputed = make_split(n, val_size, test_size, seed)
        if not np.array_equal(recomputed.train_idx, split.train_idx):
            warnings.warn(f"This scikit-learn version would produce a different split; "
                          f"using the saved split in {path}.")
        return split

    split = make_split(n, val_size, test_size, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=split.train_idx, val_idx=split.val_idx, test_idx=split.test_idx,
             n=n, val_size=val_size, test_size=test_size, split_seed=seed)
    return split


def load_train_val(cfg: dict, root: Path) -> TrainValData:
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape == (N_SAMPLES, N_FEATURES) and y.shape == (N_SAMPLES,)

    dc = cfg["data"]
    split = load_or_create_split(Path(root) / dc["splits_file"], N_SAMPLES,
                                 dc["val_size"], dc["test_size"], dc["split_seed"])

    # Fit the scaler on training rows ONLY, then apply it to validation (no leakage).
    X_train_raw = X[split.train_idx]
    scaler = StandardScaler().fit(X_train_raw)
    assert scaler.n_samples_seen_ == len(split.train_idx)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X[split.val_idx])
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-8)

    # Target stays on its original scale (median house value, $100k), so RMSE is
    # directly interpretable. Shape (n, 1) matches the model output exactly.
    return TrainValData(
        X_train=X_train.astype(np.float32),
        y_train=y[split.train_idx].reshape(-1, 1).astype(np.float32),
        X_val=X_val.astype(np.float32),
        y_val=y[split.val_idx].reshape(-1, 1).astype(np.float32),
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
    )
